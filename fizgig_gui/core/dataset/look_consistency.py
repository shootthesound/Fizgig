import json
import os
import threading

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from face_utils import FaceEmbedder
from fizgig_gui.core.config.constants import COLORS, FONT_FAMILY


class LookConsistencyMixin:
    _FF_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

    def _ff_verdict(self, sim):
        """(label, color, blurb) for a similarity score. Thresholds are ArcFace cosine-sim
        conventions: same person across varied photos usually lands 0.30-0.70 vs a single
        baseline; a different person rarely clears 0.25."""
        if sim is None:
            return ("no face", "#7F8C8D",
                    "No face detected — can't be scored. Back shots are fine; judge by eye.")
        if sim >= 0.45:
            return ("match", "#70AD47", "Solid match to the baseline look.")
        if sim >= 0.30:
            return ("borderline", "#E67E22",
                    "Same person territory, but drifting — worth an eyeball.")
        return ("drift", "#E74C3C",
                "Weak match — likely off-look (synthetic drift or a different subject).")

    def _open_face_filter_window(self):
        """Look Consistency Filter — score every training image's face against 3 chosen baselines
        (ArcFace embeddings averaged, CPU), mark drifters by click or auto-suggest, and move the
        marked ones to <folder>/excluded_by_look/. The Image Prep card places this LAST on
        purpose: it scores the finished dataset, so run it after resize/crop/captioning."""
        win = getattr(self, "_ff_win", None)
        if win is not None and win.winfo_exists():
            win.lift()
            return
        win = tk.Toplevel(self.master)
        win.title("Look Consistency Filter — face embedding drift")
        win.geometry("860x720")
        win.configure(bg=COLORS["bg_deep"])
        self._ff_win = win
        self._ff_thumbs = {}          # path -> PhotoImage (kept alive)
        self._ff_scores = {}          # path -> float similarity or None (no face)
        self._ff_marked = set()       # paths marked for exclusion
        self._ff_row_ui = {}          # path -> row widgets (in-place repaints)
        self._ff_baselines = []       # exactly 3 baseline image paths (scores are averaged)
        self._ff_busy = False
        if not hasattr(self, "_ff_embed_cache"):
            self._ff_embed_cache = {}     # (path, mtime) -> embedding or None; survives reopens
        if not hasattr(self, "_ff_embedder"):
            self._ff_embedder = None      # lazy FaceEmbedder; model load is the slow part

        head = tk.Frame(win, bg=COLORS["bg_deep"])
        head.pack(fill=tk.X, padx=14, pady=(12, 6))
        tk.Label(head, text="Look Consistency Filter", font=(FONT_FAMILY, 15, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_deep"]).pack(side=tk.LEFT)
        self._ff_apply_btn = ttk.Button(head, text="Move Marked Out of Dataset",
                                        command=self._ff_apply_moves, state="disabled")
        self._ff_apply_btn.pack(side=tk.RIGHT)
        self._ff_suggest_btn = ttk.Button(head, text="Auto-Suggest Drift",
                                          command=self._ff_auto_suggest, state="disabled")
        self._ff_suggest_btn.pack(side=tk.RIGHT, padx=(0, 8))

        base_row = tk.Frame(win, bg=COLORS["bg_deep"])
        base_row.pack(fill=tk.X, padx=14, pady=(0, 4))
        self._ff_base_slots = []
        for _ in range(3):
            holder = tk.Frame(base_row, width=72, height=72, bg=COLORS["bg_surface"],
                              highlightbackground=COLORS["border"], highlightthickness=1)
            holder.pack_propagate(False)
            holder.pack(side=tk.LEFT, padx=(0, 4))
            lbl = tk.Label(holder, text="no\nbaseline", font=(FONT_FAMILY, 8),
                           fg=COLORS["text_muted"], bg=COLORS["bg_surface"])
            lbl.pack(expand=True)
            self._ff_base_slots.append(lbl)
        base_btns = tk.Frame(base_row, bg=COLORS["bg_deep"])
        base_btns.pack(side=tk.LEFT, padx=(10, 0))
        self._ff_base_name = tk.Label(base_btns, text="Pick the 3 images that best nail the look you "
                                                      "want (Ctrl-click to select all three at once).",
                                      font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"],
                                      bg=COLORS["bg_deep"], anchor="w")
        self._ff_base_name.pack(anchor="w")
        btns = tk.Frame(base_btns, bg=COLORS["bg_deep"])
        btns.pack(anchor="w", pady=(6, 0))
        ttk.Button(btns, text="Choose 3 Baselines…", command=self._ff_choose_baselines).pack(side=tk.LEFT)
        self._ff_scan_btn = ttk.Button(btns, text="Scan Folder", command=self._ff_scan, state="disabled")
        self._ff_scan_btn.pack(side=tk.LEFT, padx=(8, 0))

        self._ff_status = tk.Label(win, text="Each score is the AVERAGE ArcFace similarity to your 3 "
                                             "baselines — averaging cancels the angle/expression/lighting "
                                             "bias any single photo carries. Same person typically lands "
                                             "30–70% (even the baselines themselves — each is scored "
                                             "against the other two as well as itself).",
                                   font=(FONT_FAMILY, 10), fg=COLORS["text_explain"], bg=COLORS["bg_deep"],
                                   justify=tk.LEFT, anchor="w", wraplength=820)
        self._ff_status.pack(fill=tk.X, padx=14)

        holder = tk.Frame(win, bg=COLORS["bg_deep"])
        holder.pack(fill=tk.BOTH, expand=True, padx=14, pady=(6, 12))
        canvas = tk.Canvas(holder, bg=COLORS["bg_deep"], highlightthickness=0)
        vbar = ttk.Scrollbar(holder, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        rows = tk.Frame(canvas, bg=COLORS["bg_deep"])
        rows_id = canvas.create_window((0, 0), window=rows, anchor="nw")
        rows.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(rows_id, width=e.width))
        # Wheel: global router (_route_mousewheel) finds this canvas via the pointer.
        self._ff_rows = rows

    def _ff_set_status(self, text):
        win = getattr(self, "_ff_win", None)
        if win is not None and win.winfo_exists():
            self._ff_status.config(text=text)

    def _ff_choose_baselines(self):
        """Exactly 3 baselines, scored by averaging — one photo bakes its own angle/expression/
        lighting bias into every score; three cancel it out."""
        folder = self.image_folder_var.get().strip()
        paths = filedialog.askopenfilenames(
            title="Choose 3 baseline images (the look you want)",
            initialdir=folder if folder and os.path.isdir(folder) else None,
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")])
        if not paths:
            return
        if len(paths) != 3:
            messagebox.showwarning(
                "Look Filter",
                f"Pick exactly 3 baseline images (you picked {len(paths)}).\n\n"
                "Ctrl-click in the file dialog to select three. Three baselines average out "
                "the angle/expression/lighting bias any single photo carries.")
            return
        # normpath BOTH here and in the scan's file list: filedialog returns forward
        # slashes while the scan builds candidates with os.path.join (backslashes on
        # Windows), so every `p in self._ff_baselines` test was False — baselines were
        # never badged, embedded twice, and could be auto-suggested and MOVED OUT of the
        # dataset by "Move Marked".
        self._ff_baselines = [os.path.normpath(p) for p in paths]
        for slot, p in zip(self._ff_base_slots, paths):
            try:
                with Image.open(p) as im:
                    im.thumbnail((68, 68), Image.LANCZOS)
                    ph = ImageTk.PhotoImage(im)
                self._ff_thumbs[f"__baseline_{p}__"] = ph
                slot.config(image=ph, text="")
            except Exception:
                slot.config(image="", text="no\npreview")
        self._ff_base_name.config(
            text="Baselines: " + ", ".join(os.path.basename(p) for p in paths))
        self._ff_scan_btn.config(state="normal")
        if self._ff_scores:
            self._ff_set_status("Baselines changed — click Scan Folder to re-score "
                                "(cached embeddings make this fast).")

    _ff_lock = threading.Lock()   # one embed at a time: Look Filter scan + gallery scorer share the model

    def _ff_embed_cached(self, path):
        """Embedding via the (path, mtime) cache — model load + detection is the slow part,
        so re-scans and baseline swaps cost almost nothing. Self-initializing and locked:
        the Look Filter scan thread and the gallery likeness scorer share one embedder."""
        try:
            key = (path, os.path.getmtime(path))
        except OSError:
            return None
        if not hasattr(self, "_ff_embed_cache"):
            self._ff_embed_cache = {}
        if key not in self._ff_embed_cache:
            with self._ff_lock:
                if key not in self._ff_embed_cache:
                    if getattr(self, "_ff_embedder", None) is None:
                        self._ff_embedder = FaceEmbedder()
                    try:
                        self._ff_embed_cache[key] = self._ff_embedder.embed(path)
                    except Exception:
                        self._ff_embed_cache[key] = None
        return self._ff_embed_cache[key]

    def _repair_embed_pil(self, pil):
        """(embedding, bbox) for the largest face in an IN-MEMORY render — FaceEmbedder.embed
        is path-only, and the pop-out metrics score renders that never touch disk. Same lock
        and same lazily-created embedder as _ff_embed_cached, so the model loads once app-wide.

        The unpadded detection gives both the embedding and a usable bbox. The pad-retry
        fallback (frame-filling faces) returns coordinates in padded-image space, so it
        contributes the embedding only — texture then measures the whole frame."""
        if FaceEmbedder is None:
            return None, None
        try:
            import cv2
            import numpy as np
            bgr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
            with self._ff_lock:
                if getattr(self, "_ff_embedder", None) is None:
                    self._ff_embedder = FaceEmbedder()
                self._ff_embedder._ensure_loaded()
                faces = self._ff_embedder._app.get(bgr)
                if faces:
                    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                    x1, y1, x2, y2 = (int(v) for v in f.bbox)
                    h, w = bgr.shape[:2]
                    bbox = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
                    return np.asarray(f.normed_embedding, dtype=np.float32), bbox
                faces = self._ff_embedder._detect_with_pad_retry(bgr)
                if faces:
                    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                    return np.asarray(f.normed_embedding, dtype=np.float32), None
        except Exception:
            pass
        return None, None

    def _ff_scan(self):
        if self._ff_busy:
            return
        folder = self.image_folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("Look Filter", "Set your training image folder on the Start tab first.")
            return
        if len(self._ff_baselines) != 3 or not all(os.path.exists(b) for b in self._ff_baselines):
            messagebox.showwarning("Look Filter", "Choose 3 baseline images first.")
            return
        # normpath to match self._ff_baselines (see the baseline picker) — the folder string
        # itself can carry forward slashes from a file dialog.
        files = sorted(
            os.path.normpath(os.path.join(folder, f)) for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
            and os.path.splitext(f)[1].lower() in self._FF_EXTS)
        if not files:
            messagebox.showinfo("Look Filter", "No images found in the training folder.")
            return
        self._ff_busy = True
        self._ff_scan_btn.config(state="disabled")
        self._ff_suggest_btn.config(state="disabled")
        self._ff_apply_btn.config(state="disabled")
        self._ff_set_status("Loading face model (first run downloads ~300 MB)…")
        win = self._ff_win

        def work():
            import numpy as np
            try:
                base_embs = [self._ff_embed_cached(b) for b in self._ff_baselines]
                missing = [os.path.basename(b) for b, e in zip(self._ff_baselines, base_embs)
                           if e is None]
                if missing:
                    self.master.after(0, lambda: self._ff_scan_done(
                        None, "No face found in baseline(s): " + ", ".join(missing) +
                        " — pick images with a clear face."))
                    return
                scores = {}
                for i, p in enumerate(files, 1):
                    if not win.winfo_exists():
                        return   # window closed mid-scan — drop the work silently
                    emb = self._ff_embed_cached(p)
                    # Average of the 3 similarities == similarity to the (unnormalized) centroid
                    # of the baselines — one photo's framing bias can't dominate the score.
                    scores[p] = None if emb is None else float(
                        np.mean([np.dot(be, emb) for be in base_embs]))
                    if i % 3 == 0 or i == len(files):
                        done, total = i, len(files)
                        self.master.after(0, lambda d=done, t=total:
                                          self._ff_set_status(f"Scoring faces… {d}/{t}"))
                self.master.after(0, lambda: self._ff_scan_done(scores, None))
            except Exception as e:
                self.master.after(0, lambda err=str(e): self._ff_scan_done(None, f"Scan failed: {err}"))

        threading.Thread(target=work, daemon=True).start()

    def _ff_scan_done(self, scores, error):
        self._ff_busy = False
        win = getattr(self, "_ff_win", None)
        if win is None or not win.winfo_exists():
            return
        self._ff_scan_btn.config(state="normal")
        if error:
            self._ff_set_status(error)
            return
        self._ff_scores = scores
        self._ff_marked &= set(scores)   # drop marks for files that vanished
        self._ff_suggest_btn.config(state="normal")
        scored = [s for s in scores.values() if s is not None]
        nf = sum(1 for s in scores.values() if s is None)
        # Persist for the trainer's "Warm up look outliers" toggle — the scores travel with the
        # dataset (same pattern as fizgig_excluded.json). Cutoff = the auto-suggest fence.
        try:
            folder = self.image_folder_var.get().strip()
            ss = sorted(scored)
            cutoff = None
            if len(ss) >= 4:
                n = len(ss)
                med, q1, q3 = ss[n // 2], ss[n // 4], ss[(3 * n) // 4]
                cutoff = max(med - 1.5 * (q3 - q1), 0.25)
            payload = {"baselines": [os.path.basename(b) for b in self._ff_baselines],
                       "cutoff": cutoff,
                       "scores": {os.path.splitext(os.path.basename(p))[0]: s
                                  for p, s in scores.items()}}
            with open(os.path.join(folder, "fizgig_look_scores.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass
        self._ff_set_status(f"{len(scores)} image(s) scored — {len(scored)} with faces, {nf} without. "
                            "Worst matches first. Click a row (or its button) to mark it, or use "
                            "Auto-Suggest. Scores saved with the dataset — low scorers you KEEP can "
                            "ease in gently: tick “Warm up look outliers” on the Training tab (Krea 2).")
        self._ff_build_rows()

    def _ff_toggle(self, path):
        if path in self._ff_marked:
            self._ff_marked.discard(path)
        else:
            self._ff_marked.add(path)
        # In-place row update — a full rebuild on every click froze the window for a beat
        # and yanked the scroll position back to the top.
        self._ff_update_row(path)
        self._ff_apply_btn.config(state="normal" if self._ff_marked else "disabled")

    def _ff_update_row(self, path):
        """Repaint one row's marked/unmarked state without rebuilding the list."""
        ui = getattr(self, "_ff_row_ui", {}).get(path)
        if ui is None:
            return
        try:
            if not ui["frame"].winfo_exists():
                return
            marked = path in self._ff_marked
            ui["frame"].config(highlightbackground="#C0392B" if marked else ui["color"],
                               highlightthickness=3 if marked else 2)
            ui["mark"].config(text="  ❌ marked for exclusion" if marked else "")
            ui["btn"].config(text="Keep" if marked else "Mark")
        except Exception:
            pass

    def _ff_auto_suggest(self):
        """Mark statistical drift: below the dataset's own low outlier fence (median − 1.5·IQR)
        or below the 0.25 different-person floor. No-face rows are never suggested — unscoreable
        isn't the same as bad (think from-behind shots)."""
        scored = sorted(s for s in self._ff_scores.values() if s is not None)
        if len(scored) < 4:
            messagebox.showinfo("Look Filter", "Not enough scored faces for statistics — mark by eye.")
            return
        n = len(scored)
        med = scored[n // 2]
        q1, q3 = scored[n // 4], scored[(3 * n) // 4]
        cutoff = max(med - 1.5 * (q3 - q1), 0.25)   # dataset's low outlier fence, floored at the
        newly = {p for p, s in self._ff_scores.items()   # ~different-person similarity level
                 if s is not None and s < cutoff and p not in self._ff_baselines}
        self._ff_marked |= newly
        self._ff_set_status(f"Auto-suggest marked {len(newly)} image(s) "
                            f"(dataset median {med * 100:.0f}%, cutoff {cutoff * 100:.0f}%). "
                            "Review before moving — it flags statistical drift, not certainty.")
        for p in newly:
            self._ff_update_row(p)
        self._ff_apply_btn.config(state="normal" if self._ff_marked else "disabled")

    def _ff_build_rows(self):
        win = getattr(self, "_ff_win", None)
        if win is None or not win.winfo_exists():
            return
        for w in self._ff_rows.winfo_children():
            w.destroy()
        self._ff_row_ui = {}   # path -> widgets for in-place mark/unmark repaints
        thumb_jobs = []
        # Worst match first; unscoreable (no face) at the bottom — they're a judgement call.
        items = sorted(self._ff_scores.items(),
                       key=lambda kv: (kv[1] is None, kv[1] if kv[1] is not None else 0.0))
        for path, sim in items:
            label, color, blurb = self._ff_verdict(sim)
            marked = path in self._ff_marked
            row = tk.Frame(self._ff_rows, bg=COLORS["bg_surface"],
                           highlightbackground="#C0392B" if marked else color,
                           highlightthickness=3 if marked else 2)
            row.pack(fill=tk.X, pady=4)

            thumb_holder = tk.Frame(row, width=100, height=100, bg=COLORS["bg_surface"])
            thumb_holder.pack_propagate(False)
            thumb_holder.pack(side=tk.LEFT, padx=8, pady=8)
            ph = self._ff_thumbs.get(path)
            tl = tk.Label(thumb_holder, bg=COLORS["bg_surface"], cursor="hand2")
            if ph is not None:
                tl.config(image=ph)
            else:
                tl.config(text="…", font=(FONT_FAMILY, 8), fg=COLORS["text_muted"])
                thumb_jobs.append((path, tl))
            tl.pack(expand=True)
            tl.bind("<Button-1>", lambda e, p=path: self._ff_toggle(p))

            info = tk.Frame(row, bg=COLORS["bg_surface"])
            info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8), pady=8)
            name_row = tk.Frame(info, bg=COLORS["bg_surface"])
            name_row.pack(fill=tk.X, anchor="w")
            tk.Label(name_row, text=os.path.basename(path), font=(FONT_FAMILY, 10, "bold"),
                     fg=COLORS["text_primary"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT)
            tk.Label(name_row, text=f"  {label.upper()}", font=(FONT_FAMILY, 9, "bold"),
                     fg=color, bg=COLORS["bg_surface"]).pack(side=tk.LEFT)
            if path in self._ff_baselines:
                tk.Label(name_row, text="  ★ baseline", font=(FONT_FAMILY, 9),
                         fg="#F1C40F", bg=COLORS["bg_surface"]).pack(side=tk.LEFT)
            mark_lbl = tk.Label(name_row, text="  ❌ marked for exclusion" if marked else "",
                                font=(FONT_FAMILY, 9, "bold"),
                                fg="#C0392B", bg=COLORS["bg_surface"])
            mark_lbl.pack(side=tk.LEFT)
            btn = ttk.Button(name_row, text="Keep" if marked else "Mark", width=8,
                             command=lambda p=path: self._ff_toggle(p))
            btn.pack(side=tk.RIGHT, padx=(8, 0))
            self._ff_row_ui[path] = {"frame": row, "mark": mark_lbl, "btn": btn, "color": color}

            sim_txt = f"match {sim * 100:.0f}%" if sim is not None else "no face to score"
            tk.Label(info, text=sim_txt, font=(FONT_FAMILY, 9),
                     fg=COLORS["text_secondary"], bg=COLORS["bg_surface"]).pack(anchor="w", pady=(2, 0))
            tk.Label(info, text=blurb, font=(FONT_FAMILY, 8, "italic"),
                     fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(anchor="w", pady=(2, 0))
        if thumb_jobs:
            self._load_thumbs_async(thumb_jobs, self._ff_thumbs)
        self._ff_apply_btn.config(state="normal" if self._ff_marked else "disabled")

    def _ff_apply_moves(self):
        """Move marked images (+ their caption .txt) to <folder>/excluded_by_look/. Never deletes."""
        marked = [p for p in self._ff_marked if os.path.exists(p)]
        if not marked:
            return
        folder = self.image_folder_var.get().strip()
        dest_dir = os.path.join(folder, "excluded_by_look")
        if not messagebox.askyesno(
                "Move marked images",
                f"Move {len(marked)} image(s) (plus matching captions) out of the dataset "
                f"to:\n\n{dest_dir}\n\nNothing is deleted — move them back to re-admit them."):
            return
        import shutil
        os.makedirs(dest_dir, exist_ok=True)
        moved = 0
        for p in marked:
            target = None
            try:
                base = os.path.basename(p)
                target = os.path.join(dest_dir, base)
                n = 2
                while os.path.exists(target):
                    stem, ext = os.path.splitext(base)
                    target = os.path.join(dest_dir, f"{stem}_{n}{ext}")
                    n += 1
                shutil.move(p, target)
                cap = os.path.splitext(p)[0] + ".txt"
                if os.path.exists(cap):
                    cap_target = os.path.splitext(target)[0] + ".txt"
                    shutil.move(cap, cap_target)
                moved += 1
                self._ff_scores.pop(p, None)
                self._ff_marked.discard(p)
                # Drop just this row — rebuilding the whole list is slow and loses scroll position.
                ui = self._ff_row_ui.pop(p, None)
                if ui is not None:
                    try:
                        ui["frame"].destroy()
                    except Exception:
                        pass
            except Exception as e:
                # A failed move can leave a half-state behind (shutil.move falls back to
                # copy+delete when the source is briefly locked, e.g. mid thumbnail decode;
                # the copy lands, the delete fails). Remove the orphan copy so the image
                # isn't duplicated in and out of the dataset.
                try:
                    if target and os.path.exists(p) and os.path.exists(target):
                        os.remove(target)
                except Exception:
                    pass
                self._ff_set_status(f"Could not move {os.path.basename(p)}: {e}")
        self._ff_apply_btn.config(state="normal" if self._ff_marked else "disabled")
        self._ff_set_status(f"Moved {moved} image(s) to {dest_dir}. "
                            f"{len(self._ff_scores)} image(s) remain in the dataset.")