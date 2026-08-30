import os

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from fizgig_gui.core.config.constants import COLORS, FONT_FAMILY, BG_COLOR


class MetadataTabMixin:
    def create_metadata_tab(self):
        """Create the Metadata tab — view and edit the modelspec metadata on any .safetensors
        file: LoRAs, DiTs, text encoders, embeddings, whatever — including ones Fizgig didn't
        train, or trained before these fields existed."""
        scrollable_frame, _ = self.create_scrollable_frame(self.metadata_tab)

        outer = tk.Frame(scrollable_frame, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)

        self._add_tab_banner(
            outer,
            "Metadata",
            "View and edit the SAI ModelSpec metadata embedded in a .safetensors file — title, "
            "author, description, trigger phrase, thumbnail, and anything else ComfyUI's model "
            "browser reads. Works on any .safetensors file — LoRA, DiT, text encoder, "
            "embedding — not just ones Fizgig trained.",
        )

        self._metadata_editor_path = None
        self._metadata_editor_thumbnail_uri = None  # current thumbnail data URI, or None
        self._metadata_editor_custom = {}  # every key outside the standard fields below
        self._metadata_thumb_photo = None  # keeps the PhotoImage alive; Tk drops it otherwise

        # --- Load ---
        load_card = self._start_section_card(
            outer, "File",
            "Pick any .safetensors file — LoRA, DiT, text encoder, embedding — its current "
            "metadata loads below.",
        )
        load_card.grid_columnconfigure(1, weight=1)
        ttk.Label(load_card, text="File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self.metadata_file_var = tk.StringVar()
        ttk.Entry(load_card, textvariable=self.metadata_file_var, width=60).grid(
            row=0, column=1, sticky=tk.EW, pady=4)
        ttk.Button(load_card, text="Browse", command=self._browse_metadata_file).grid(
            row=0, column=2, sticky=tk.W, padx=(8, 0), pady=4)
        self._metadata_status_label = tk.Label(
            load_card, text="No file loaded.", font=(FONT_FAMILY, 9, "italic"),
            fg=COLORS["text_muted"], bg=COLORS["bg_surface"])
        self._metadata_status_label.grid(row=1, column=1, sticky=tk.W, pady=(2, 0))

        # --- Standard fields ---
        fields_card = self._start_section_card(
            outer, "Standard Fields",
            "The fields ComfyUI's model browser (and other spec-aware tools) render specially.",
        )
        fields_card.grid_columnconfigure(1, weight=1)

        def _field_row(label_text, row):
            ttk.Label(fields_card, text=label_text).grid(
                row=row, column=0, sticky=tk.NW, padx=(0, 10), pady=4)
            var = tk.StringVar()
            ttk.Entry(fields_card, textvariable=var, width=60).grid(
                row=row, column=1, sticky=tk.EW, pady=4)
            return var

        self.metadata_title_var = _field_row("Title:", 0)
        self.metadata_author_var = _field_row("Author:", 1)
        self.metadata_license_var = _field_row("License:", 2)
        self.metadata_tags_var = _field_row("Tags:", 3)
        self.metadata_trigger_var = _field_row("Trigger Phrase:", 4)
        self.metadata_usage_hint_var = _field_row("Usage Hint:", 5)

        ttk.Label(fields_card, text="Description:").grid(
            row=6, column=0, sticky=tk.NW, padx=(0, 10), pady=4)
        self.metadata_description_text = tk.Text(
            fields_card, width=60, height=5, wrap=tk.WORD,
            bg=COLORS["bg_surface"], fg=COLORS["text_primary"],
            insertbackground=COLORS["text_primary"], font=(FONT_FAMILY, 10),
            relief="flat", highlightthickness=1,
            highlightbackground=COLORS["border"], highlightcolor=COLORS["border_focus"],
        )
        self.metadata_description_text.grid(row=6, column=1, sticky=tk.EW, pady=4)

        # --- Thumbnail ---
        thumb_card = self._start_section_card(
            outer, "Thumbnail",
            "The image ComfyUI shows as card art. Fizgig auto-embeds the latest training sample "
            "when it trains a LoRA — replace or clear it here for any file.",
        )
        self._metadata_thumb_label = tk.Label(thumb_card, bg=COLORS["bg_surface"],
                                              text="(no thumbnail)", fg=COLORS["text_muted"])
        self._metadata_thumb_label.pack(anchor=tk.W, pady=(0, 8))
        thumb_btn_row = tk.Frame(thumb_card, bg=COLORS["bg_surface"])
        thumb_btn_row.pack(anchor=tk.W)
        ttk.Button(thumb_btn_row, text="Replace...",
                   command=self._browse_metadata_thumbnail).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(thumb_btn_row, text="Clear",
                   command=self._clear_metadata_thumbnail).pack(side=tk.LEFT)

        # --- Custom fields ---
        custom_card = self._start_section_card(
            outer, "Custom Fields",
            "Like ID3 tags on an MP3 — the format isn't limited to a fixed list, and a reader "
            "just ignores whatever it doesn't recognize. Add anything you want: author_email, "
            "a colorspace profile note, whatever's useful to you. Not part of the SAI ModelSpec "
            "standard, so tools other than Fizgig won't render these specially, but they're "
            "stored in the file like any other metadata. Also shows any non-standard keys "
            "already in the file — nothing gets silently dropped on save.",
        )
        tree_frame = tk.Frame(custom_card, bg=COLORS["bg_surface"])
        tree_frame.pack(fill=tk.BOTH, expand=True)
        self.metadata_custom_tree = ttk.Treeview(
            tree_frame, columns=("key", "value"), show="headings", height=6)
        self.metadata_custom_tree.heading("key", text="Key")
        self.metadata_custom_tree.heading("value", text="Value")
        self.metadata_custom_tree.column("key", width=220, anchor=tk.W)
        self.metadata_custom_tree.column("value", width=420, anchor=tk.W)
        self.metadata_custom_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL,
                                    command=self.metadata_custom_tree.yview)
        tree_scroll.pack(side=tk.LEFT, fill=tk.Y)
        self.metadata_custom_tree.configure(yscrollcommand=tree_scroll.set)

        custom_btn_row = tk.Frame(custom_card, bg=COLORS["bg_surface"])
        custom_btn_row.pack(anchor=tk.W, pady=(8, 0))
        ttk.Button(custom_btn_row, text="Add field...",
                   command=self._add_metadata_custom_field).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(custom_btn_row, text="Remove selected",
                   command=self._remove_metadata_custom_field).pack(side=tk.LEFT)

        # --- Save ---
        save_row = tk.Frame(outer, bg=COLORS["bg_deep"])
        save_row.pack(fill=tk.X, padx=36, pady=(0, 20))
        ttk.Button(save_row, text="Save",
                   command=lambda: self._save_metadata_file(save_as=False)).pack(
            side=tk.LEFT, padx=(0, 8))
        ttk.Button(save_row, text="Save As...",
                   command=lambda: self._save_metadata_file(save_as=True)).pack(side=tk.LEFT)
        self._metadata_save_status = tk.Label(
            save_row, text="", font=(FONT_FAMILY, 9, "italic"),
            fg=COLORS["text_muted"], bg=COLORS["bg_deep"])
        self._metadata_save_status.pack(side=tk.LEFT, padx=(16, 0))

    def _browse_metadata_file(self):
        filepath = filedialog.askopenfilename(
            title="Select a .safetensors file",
            filetypes=[("SafeTensors", "*.safetensors"), ("All files", "*.*")],
            initialdir=self._lora_initialdir(),
        )
        if filepath:
            self.metadata_file_var.set(filepath)
            self._load_metadata_file(filepath)

    def _load_metadata_file(self, path):
        from fizgig.training.metadata import load_metadata_from_safetensors
        try:
            meta = load_metadata_from_safetensors(path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read metadata:\n{e}")
            return

        self._metadata_editor_path = path
        standard = {
            "modelspec.title": self.metadata_title_var,
            "modelspec.author": self.metadata_author_var,
            "modelspec.license": self.metadata_license_var,
            "modelspec.tags": self.metadata_tags_var,
            "modelspec.trigger_phrase": self.metadata_trigger_var,
            "modelspec.usage_hint": self.metadata_usage_hint_var,
        }
        for key, var in standard.items():
            var.set(meta.get(key, "") or "")

        self.metadata_description_text.delete("1.0", tk.END)
        self.metadata_description_text.insert("1.0", meta.get("modelspec.description", "") or "")

        self._metadata_editor_thumbnail_uri = meta.get("modelspec.thumbnail")
        self._show_metadata_thumbnail_preview(self._metadata_editor_thumbnail_uri)

        skip_keys = set(standard.keys()) | {"modelspec.description", "modelspec.thumbnail"}
        self._metadata_editor_custom = {k: v for k, v in meta.items() if k not in skip_keys}
        self._refresh_metadata_custom_tree()

        n = len(meta)
        self._metadata_status_label.config(
            text=f"Loaded — {n} metadata key{'s' if n != 1 else ''} found.",
            fg=COLORS["text_secondary"])
        self._metadata_save_status.config(text="")

    def _show_metadata_thumbnail_preview(self, data_uri):
        if not data_uri or not str(data_uri).startswith("data:image"):
            self._metadata_thumb_label.config(image="", text="(no thumbnail)")
            self._metadata_thumb_photo = None
            return
        try:
            import base64
            from io import BytesIO
            b64 = data_uri.split(",", 1)[1]
            img = Image.open(BytesIO(base64.b64decode(b64)))
            img.thumbnail((256, 256))
            photo = ImageTk.PhotoImage(img)
            self._metadata_thumb_photo = photo  # reference kept alive deliberately
            self._metadata_thumb_label.config(image=photo, text="")
        except Exception:
            self._metadata_thumb_label.config(image="", text="(couldn't decode thumbnail)")
            self._metadata_thumb_photo = None

    def _browse_metadata_thumbnail(self):
        filepath = filedialog.askopenfilename(
            title="Select a thumbnail image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp"), ("All files", "*.*")],
        )
        if not filepath:
            return
        from fizgig.training.metadata import thumbnail_data_uri
        uri = thumbnail_data_uri(filepath)
        if not uri:
            messagebox.showerror("Error", "Could not read that image.")
            return
        self._metadata_editor_thumbnail_uri = uri
        self._show_metadata_thumbnail_preview(uri)

    def _clear_metadata_thumbnail(self):
        self._metadata_editor_thumbnail_uri = None
        self._show_metadata_thumbnail_preview(None)

    def _refresh_metadata_custom_tree(self):
        self.metadata_custom_tree.delete(*self.metadata_custom_tree.get_children())
        for k, v in sorted(self._metadata_editor_custom.items()):
            display_v = v if len(str(v)) <= 120 else str(v)[:117] + "..."
            self.metadata_custom_tree.insert("", tk.END, iid=k, values=(k, display_v))

    def _add_metadata_custom_field(self):
        dlg = tk.Toplevel(self.master)
        dlg.title("Add custom field")
        dlg.configure(bg=BG_COLOR)
        dlg.transient(self.master)
        tk.Label(dlg, text="Key:", bg=BG_COLOR, fg=COLORS["text_secondary"]).grid(
            row=0, column=0, sticky=tk.W, padx=10, pady=(10, 2))
        key_entry = ttk.Entry(dlg, width=40)
        key_entry.grid(row=0, column=1, padx=10, pady=(10, 2))
        tk.Label(dlg, text="Value:", bg=BG_COLOR, fg=COLORS["text_secondary"]).grid(
            row=1, column=0, sticky=tk.W, padx=10, pady=2)
        val_entry = ttk.Entry(dlg, width=40)
        val_entry.grid(row=1, column=1, padx=10, pady=2)

        def ok():
            k = key_entry.get().strip()
            v = val_entry.get().strip()
            if k:
                self._metadata_editor_custom[k] = v
                self._refresh_metadata_custom_tree()
            dlg.destroy()

        btn_row = tk.Frame(dlg, bg=BG_COLOR)
        btn_row.grid(row=2, column=0, columnspan=2, pady=10)
        ttk.Button(btn_row, text="Cancel", command=dlg.destroy).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="Add", command=ok).pack(side=tk.LEFT, padx=5)
        key_entry.bind("<Return>", lambda e: ok())
        key_entry.focus_set()
        dlg.grab_set()

    def _remove_metadata_custom_field(self):
        for k in self.metadata_custom_tree.selection():
            self._metadata_editor_custom.pop(k, None)
        self._refresh_metadata_custom_tree()

    def _save_metadata_file(self, save_as=False):
        if not self._metadata_editor_path:
            messagebox.showinfo("No file loaded", "Load a .safetensors file first.")
            return

        dest = self._metadata_editor_path
        if save_as:
            dest = filedialog.asksaveasfilename(
                title="Save file as",
                defaultextension=".safetensors",
                filetypes=[("SafeTensors", "*.safetensors")],
                initialfile=os.path.basename(self._metadata_editor_path),
            )
            if not dest:
                return
        elif not messagebox.askyesno(
                "Overwrite?",
                f"Save metadata changes to:\n{dest}\n\nThis overwrites the file in place."):
            return

        try:
            from safetensors.torch import load_file, save_file
            # .clone() forces a real copy out of the mmap'd view load_file returns. Without
            # this, saving back onto the SAME path fails on Windows with "the requested
            # operation cannot be performed on a file with a user-mapped section open"
            # (error 1224) — the mapping from this exact load is still active, and Windows
            # (unlike Linux) refuses to overwrite a file while it's mapped. Cloning breaks
            # that dependency before we ever try to write.
            tensors = {k: v.clone() for k, v in load_file(self._metadata_editor_path).items()}
        except Exception as e:
            messagebox.showerror("Error", f"Could not read the file's tensors:\n{e}")
            return

        new_meta = dict(self._metadata_editor_custom)
        field_map = {
            "modelspec.title": self.metadata_title_var.get().strip(),
            "modelspec.author": self.metadata_author_var.get().strip(),
            "modelspec.license": self.metadata_license_var.get().strip(),
            "modelspec.tags": self.metadata_tags_var.get().strip(),
            "modelspec.trigger_phrase": self.metadata_trigger_var.get().strip(),
            "modelspec.usage_hint": self.metadata_usage_hint_var.get().strip(),
            "modelspec.description": self.metadata_description_text.get("1.0", "end-1c").strip(),
        }
        for k, v in field_map.items():
            if v:
                new_meta[k] = v
        if self._metadata_editor_thumbnail_uri:
            new_meta["modelspec.thumbnail"] = self._metadata_editor_thumbnail_uri

        # A metadata-only edit still goes through a full resave, which isn't guaranteed to be
        # byte-identical to the original — so hashes computed over the old bytes can no longer
        # be trusted. Same move bake.py already makes whenever it changes a file's contents.
        for stale in ("sshs_model_hash", "sshs_legacy_hash", "modelspec.hash_sha256"):
            new_meta.pop(stale, None)

        # Write to a temp file and swap it in atomically, so a failed/interrupted save can
        # never leave the original file half-written.
        tmp_dest = dest + ".tmp"
        try:
            save_file(tensors, tmp_dest, metadata=new_meta)
            os.replace(tmp_dest, dest)
        except Exception as e:
            try:
                if os.path.exists(tmp_dest):
                    os.remove(tmp_dest)
            except OSError:
                pass
            messagebox.showerror("Error", f"Could not save:\n{e}")
            return

        self._metadata_save_status.config(text=f"Saved {os.path.basename(dest)}",
                                          fg=COLORS["text_secondary"])
        if save_as:
            self.metadata_file_var.set(dest)
            self._metadata_editor_path = dest

