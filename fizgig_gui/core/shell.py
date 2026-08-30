import json
import os
import subprocess
import sys

import tkinter as tk
from tkinter import messagebox, ttk

from fizgig_gui.core.config.constants import COLORS, FONT_FAMILY, SAMPLE_RESOLUTIONS
from fizgig_gui.core.config.last_used import save_last_used
from fizgig_gui.core.config.prefs import _persist_disabled, save_prefs


class ShellMixin:
    def _append_global_log(self, text):
        """Append text to the global log buffer and push to popup if open.

        Thread-safe by marshalling: several workers (profiler, extract, engine loads) log
        from their own threads, and writing the console popup's Text widget off the main
        thread is a Tcl panic — a hard process crash no try/except can catch. That is how
        clicking the IDLE/BUSY light during a model load killed the app: the click queued,
        the popup opened as the load returned, and the next worker log write hit it."""
        import threading
        if threading.current_thread() is not threading.main_thread():
            self.master.after(0, self._append_global_log, text)
            return
        self._log_buffer.append(text)
        if len(self._log_buffer) > 50000:
            self._log_buffer = self._log_buffer[-40000:]
        if self._console_popup_text is not None:
            try:
                at_bottom = self._console_popup_text.yview()[1] >= 0.999
                self._console_popup_text.configure(state="normal")
                self._console_popup_text.insert(tk.END, text)
                if at_bottom:
                    self._console_popup_text.see(tk.END)
                self._console_popup_text.configure(state="disabled")
            except Exception:
                pass

    def _royale_is_busy(self):
        """True while any LoRA Royale work is in flight (epoch render / seed / prompt /
        strength travel / morph export / likeness scoring)."""
        return any(getattr(self, f, False) for f in (
            '_royale_rendering', '_royale_traveling', '_royale_pt_running',
            '_royale_lora_running', '_royale_exporting', '_royale_scoring',
            '_royale_cmp_running'))

    @staticmethod
    def _wheel_units(event):
        """Scroll units from either wheel encoding: Windows <MouseWheel> delta (±120 per
        notch, high-res mice send smaller values) or X11 Button-4/5 (pods)."""
        num = getattr(event, "num", 0)
        if num == 4:
            return -3
        if num == 5:
            return 3
        d = getattr(event, "delta", 0)
        if not d:
            return 0
        return int(-1 * (d / 120)) if abs(d) >= 120 else (-1 if d > 0 else 1)

    def _route_mousewheel(self, event):
        """Global wheel dispatch — see the install site for the routing rules."""
        try:
            w = self.master.winfo_containing(event.x_root, event.y_root)
        except Exception:
            w = None
        if w is None:
            return
        units = self._wheel_units(event)
        if not units:
            return
        node, hops = w, 0
        while node is not None and hops < 40:
            # Native scrollers own the wheel while hovered — their class bindings already
            # scroll them, and routing the page underneath as well would double-scroll.
            if isinstance(node, (tk.Text, tk.Listbox, ttk.Treeview)):
                return
            if isinstance(node, tk.Canvas):
                try:
                    if node.cget("yscrollcommand"):
                        node.yview_scroll(units, "units")
                        return "break"
                except tk.TclError:
                    pass
            node = getattr(node, "master", None)
            hops += 1
        return

    def _wheel_over_dropdown(self, event):
        """Wheel over a Combobox/Spinbox: scroll the page, never the value."""
        self._route_mousewheel(event)
        return "break"

    def _is_render_busy(self):
        """In-process GPU render on a tab that unloads its engine on switch (Repair
        Studio / Explorer / Royale). Switching tabs here would reset a busy engine and
        hang the app — used to lock tab switching. Excludes the training subprocess
        (separate process; switching during a run is fine)."""
        return (getattr(self, '_repair_preview_in_flight', False)
                or getattr(self, '_explorer_generating', False)
                or self._royale_is_busy())

    def _is_any_busy(self):
        """Return True if any background work is in progress."""
        if self.current_process is not None:
            try:
                if self.current_process.poll() is None:
                    return True
            except Exception:
                pass
        if getattr(self, '_captioning_running', False):
            return True
        if getattr(self, '_translating', False):
            return True
        if getattr(self, '_fetch_running', False):
            return True   # model download in flight — a queued run must not read partial files
        if getattr(self, '_repair_preview_in_flight', False):
            return True
        if getattr(self, '_explorer_generating', False):
            return True
        if self._royale_is_busy():
            return True
        # Profiler running (button disabled while active)
        if hasattr(self, 'profiler_run_btn'):
            try:
                if str(self.profiler_run_btn.cget("state")) == "disabled":
                    return True
            except Exception:
                pass
        # Extractor running
        if hasattr(self, 'extract_run_btn'):
            try:
                if str(self.extract_run_btn.cget("state")) == "disabled":
                    return True
            except Exception:
                pass
        return False

    def _update_status_indicator(self):
        """Poll busy state and redraw the IDLE/BUSY 'studio light': a lit circle
        with a soft glow + all-caps word + a matching-colour frame, all in the
        status colour (green idle / red busy)."""
        # Lock tab switching during an in-process render — switching can unload a busy
        # engine and hang the app. (Training is a subprocess, so it doesn't lock.)
        try:
            render_busy = self._is_render_busy()
            if render_busy != getattr(self, "_tabs_locked", None):
                self._tabs_locked = render_busy
                cur = self.notebook.select()
                for tid in self.notebook.tabs():
                    self.notebook.tab(tid, state=("disabled" if (render_busy and tid != cur) else "normal"))
        except Exception:
            pass
        try:
            busy = self._is_any_busy()
            color = COLORS["error"] if busy else COLORS["success"]
            label = "BUSY" if busy else "IDLE"
            bg = COLORS["bg_deep"]
            c = self._status_canvas
            c.delete("all")
            w = int(c["width"]); h = int(c["height"])
            cy = h // 2
            dx = 13
            d = 9
            # soft glow: concentric rings fading from the background up to the
            # status colour (drawn outer→inner so the brightest sits nearest).
            for gd, t in ((d + 12, 0.16), (d + 8, 0.32), (d + 4, 0.55)):
                c.create_oval(dx - gd / 2, cy - gd / 2, dx + gd / 2, cy + gd / 2,
                              fill=self._lerp_color(bg, color, t), outline="")
            # outer frame (the warning-light surround)
            c.create_rectangle(1, 1, w - 1, h - 1, outline=color, width=2)
            # the lit dot
            c.create_oval(dx - d / 2, cy - d / 2, dx + d / 2, cy + d / 2,
                          fill=color, outline=color)
            # the word — centred in the gap between the dot's right edge and the
            # right edge of the frame
            text_cx = ((dx + d / 2) + (w - 2)) / 2
            c.create_text(text_cx, cy + 1, text=label, anchor="center",
                          fill=color, font=(FONT_FAMILY, 9, "bold"))
        except Exception:
            pass
        self.master.after(500, self._update_status_indicator)

    def _open_console_popup(self):
        """Open (or raise) the console log popup window."""
        if self._console_popup is not None:
            try:
                if self._console_popup.winfo_exists():
                    self._console_popup.lift()
                    return
            except Exception:
                pass
            self._console_popup = None

        win = tk.Toplevel(self.master)
        win.title("Fizgig — Console Log")
        win.geometry("900x500")
        win.minsize(400, 200)
        win.configure(bg=COLORS["bg_deep"])

        text = tk.Text(win, bg=COLORS["bg_deep"], fg=COLORS["text_primary"],
                       font=("Consolas", 9), wrap="word", state="disabled",
                       selectbackground=COLORS["accent"], borderwidth=0,
                       padx=12, pady=8)
        scrollbar = ttk.Scrollbar(win, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Populate with existing history
        text.configure(state="normal")
        text.insert(tk.END, "".join(self._log_buffer))
        text.see(tk.END)
        text.configure(state="disabled")

        self._console_popup = win
        self._console_popup_text = text

        def _on_close():
            self._console_popup = None
            self._console_popup_text = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_close)

    def _on_caption_folder_changed(self, *args):
        """Reset caption images loaded flag when folder changes"""
        self.caption_images_loaded = False

    def on_tab_changed(self, event):
        """Handle notebook tab changes"""
        selected_tab = self.notebook.select()
        tab_text = self.notebook.tab(selected_tab, "text")

        # When Captions tab is selected, load images if folder is set
        if tab_text == "3. Captions":
            folder = self.image_folder_var.get()
            if folder and os.path.isdir(folder) and not self.caption_images_loaded:
                self.refresh_caption_images()
                self.caption_images_loaded = True

        # When leaving Repair Studio / Explorer / Royale, unload pipeline to free VRAM —
        # but NEVER unload an engine that's mid-render on a worker thread (resetting it
        # under the worker hard-hangs the app). Leave it loaded; it frees on a later
        # idle switch. Tab switching is also locked while rendering (see status poll),
        # so this is a belt-and-suspenders guard.
        if tab_text != "Repair Studio" and not getattr(self, '_repair_preview_in_flight', False):
            self._unload_repair_studio_models()
        if tab_text != "LoRA the Explorer" and not getattr(self, '_explorer_generating', False):
            self._unload_explorer_models()
        if tab_text != "LoRA Royale" and not self._royale_is_busy():
            self._royale_unload()

        # Entering a heavy-engine tab releases the warm caption worker (~8 GB Qwen): those
        # engines are 10-20 GB each and plan against free VRAM at load. Guarded on a job in
        # flight, like every other unload here. Elsewhere the worker deliberately stays warm
        # (fast Regenerate) until Unload / Start Training / app close.
        if (tab_text in ("Repair Studio", "LoRA the Explorer", "LoRA Royale")
                and self._caption_worker_alive()
                and not getattr(self, "_captioning_running", False)):
            self.update_caption_log("Caption model released (freeing VRAM for "
                                    f"{tab_text}).\n")
            self._stop_caption_worker_async(lambda: None, graceful=False)

    def remove_focus(self, event):
        """Remove focus from active widget when clicking background"""
        self.master.focus_set()

    def _open_in_file_manager(self, path: str):
        """Open a file or folder in the OS's native file manager.

        Uses os.startfile on Windows, `open` on macOS, and `xdg-open` on Linux.
        `os.name == 'posix'` matches both macOS and Linux, so we fall back to
        sys.platform for the Mac/Linux split.
        """
        try:
            if os.name == 'nt':
                os.startfile(path)
            elif sys.platform == 'darwin':
                subprocess.run(['open', path])
            else:
                subprocess.run(['xdg-open', path])
        except Exception as e:
            messagebox.showerror("Open Failed", f"Could not open:\n{path}\n\n{e}")

    # Setup-area fields remembered across restarts for the two hands-on workbench tabs.
    # (attr, last_used key) — one table drives restore, the save stanza, and the traces.
    _WORKBENCH_REMEMBER = [
        ("repair_primary_var", "repair_primary"),
        ("repair_donor_var", "repair_donor"),
        ("repair_prompt_var", "repair_prompt"),
        ("repair_seed_var", "repair_seed"),
        ("repair_res_var", "repair_res"),
        ("repair_turbo_var", "repair_turbo"),
        ("repair_ref_path_var", "repair_ref_path"),
        ("repair_ref_mp_var", "repair_ref_mp"),
        ("repair_ref_strength_var", "repair_ref_strength"),
        ("repair_metrics_ref_var", "repair_metrics_ref"),
        ("explorer_lora_var", "explorer_lora"),
        ("explorer_prompt_var", "explorer_prompt"),
        ("explorer_ref_path_var", "explorer_ref_path"),
        ("explorer_ref_mp_var", "explorer_ref_mp"),
        ("explorer_ref_strength_var", "explorer_ref_strength"),
        ("explorer_seed_var", "explorer_seed"),
        ("explorer_res_var", "explorer_res"),
        ("explorer_intensity_var", "explorer_intensity"),
        ("explorer_mutations_var", "explorer_mutations"),
        ("explorer_structure_var", "explorer_structure"),
    ]

    def _restore_workbench_setup_fields(self):
        """Restore remembered Repair Studio / Explorer Setup values, then attach debounced
        save traces so edits persist without needing a clean app close. Restore is safe at
        startup: every var trace on these fields no-ops while its engine is unloaded."""
        for attr, key in self._WORKBENCH_REMEMBER:
            var = getattr(self, attr, None)
            if var is None or key not in self.last_used:
                continue
            try:
                var.set(self.last_used[key])
            except Exception:
                pass
        # Traces AFTER restore, so restoring doesn't immediately rewrite the file N times.
        self._workbench_save_after = None

        def _debounced_save(*_):
            if self._workbench_save_after is not None:
                try:
                    self.master.after_cancel(self._workbench_save_after)
                except Exception:
                    pass
            self._workbench_save_after = self.master.after(600, self._save_last_used_paths)

        for attr, _key in self._WORKBENCH_REMEMBER:
            var = getattr(self, attr, None)
            if var is not None:
                try:
                    var.trace_add("write", _debounced_save)
                except Exception:
                    pass

    def _save_last_used_paths(self, *args):
        """Save last-used folder paths and settings to config file"""
        if _persist_disabled():
            return
        # Seed from what's already remembered, THEN overwrite with live widget values. Keys that
        # are set directly on self.last_used (the per-tab Klein/Krea 2 family selectors, for
        # instance) have no widget stanza below, so building `data` from scratch silently dropped
        # them on the very next save — the setting looked persisted until you restarted.
        data = dict(self.last_used) if isinstance(getattr(self, "last_used", None), dict) else {}
        # Caption model / task / token budget — these were read at widget-creation time and never
        # written back, so every restart reset them. Now that the model choice matters (Florence
        # vs Qwen3-VL), losing it silently would be a real annoyance.
        for _attr, _key in (("caption_model_var", "caption_model"),
                            ("caption_task_var", "caption_task"),
                            ("caption_max_tokens_var", "caption_max_tokens")):
            _v = getattr(self, _attr, None)
            if _v is not None:
                try:
                    data[_key] = _v.get()
                except Exception:
                    pass
        # Per-model task memory. The flat "caption_task" above is still written so an older
        # build (or a downgrade) finds something sensible rather than nothing.
        if getattr(self, "_caption_task_memory", None):
            data["caption_tasks"] = dict(self._caption_task_memory)
        data.update({
            "prep_mode": self.prep_mode_var.get(),
            "prep_replace_originals": bool(self.delete_originals_var.get()),
            "prep_megapixels": self.prep_megapixels_var.get(),
            "image_folder": self.image_folder_var.get(),
            "image_folder2": (self._concept_folder_vars[0].get()
                              if getattr(self, "_concept_folder_vars", None) else ""),
            "caption_trigger": self.caption_text_var.get(),
            "dataset_cache_dir": self.dataset_cache_dir_var.get(),
        })
        # Save architecture if variable exists
        if hasattr(self, 'architecture_var'):
            data["architecture"] = self.architecture_var.get()
        # Save sample prompt if widget exists
        if hasattr(self, 'sample_prompt_text'):
            data["sample_prompt"] = self.sample_prompt_text.get("1.0", tk.END).strip()
        # Save the sample reference image path
        if hasattr(self, 'sample_ref_image_var'):
            data["sample_ref_image"] = self.sample_ref_image_var.get()
        if hasattr(self, 'sample_frames_var'):
            data["sample_frames"] = self.sample_frames_var.get()
        # Krea 2 preview engine (Samples tab) — stored canonical, not the display label
        if hasattr(self, 'krea2_preview_engine_var'):
            data["krea2_preview_engine"] = self._krea2_preview_engine()
        # Repair Studio / Explorer Setup fields (one shared table drives restore + save)
        for _attr, _key in self._WORKBENCH_REMEMBER:
            _var = getattr(self, _attr, None)
            if _var is not None:
                try:
                    data[_key] = _var.get()
                except Exception:
                    pass
        # Save LoRA output directory if entry exists
        if "LORA_OUTPUT_DIR" in self.entries:
            data["lora_output_dir"] = self.entries["LORA_OUTPUT_DIR"].get()
        # Remember the last LoRA Royale checkpoint folder + render inputs
        if hasattr(self, 'royale_folder_var'):
            data["royale_folder"] = self.royale_folder_var.get()
        if hasattr(self, 'royale_mode_var'):
            data["royale_mode"] = self.royale_mode_var.get()
            data["royale_single"] = self.royale_single_var.get()
        if hasattr(self, 'royale_prompt_var'):
            data["royale_prompt"] = self.royale_prompt_var.get()
            data["royale_seed"] = self.royale_seed_var.get()
            data["royale_max"] = self.royale_max_var.get()
            data["royale_ref"] = self.royale_ref_var.get()
            data["royale_ref_strength"] = self.royale_ref_strength_var.get()
            data["royale_w"] = self.royale_w_var.get()
            data["royale_h"] = self.royale_h_var.get()
        if hasattr(self, 'royale_like_ref_var'):
            data["royale_like_ref"] = self.royale_like_ref_var.get()
        if hasattr(self, 'royale_travel_seed_a_var'):
            data["royale_travel_seed_a"] = self.royale_travel_seed_a_var.get()
            data["royale_travel_seed_b"] = self.royale_travel_seed_b_var.get()
            data["royale_travel_w"] = self.royale_travel_w_var.get()
            data["royale_travel_h"] = self.royale_travel_h_var.get()
            data["royale_travel_ref"] = self.royale_travel_ref_var.get()
            data["royale_travel_use_epoch_ref"] = bool(self.royale_travel_use_epoch_ref_var.get())
            data["royale_travel_ref_strength"] = self.royale_travel_ref_strength_var.get()
            data["royale_travel_ref_mp"] = self.royale_travel_ref_mp_var.get()
            data["royale_travel_seq_ref"] = bool(self.royale_travel_seq_ref_var.get())
            data["royale_travel_waypoints"] = self.royale_travel_waypoints_var.get()
        if hasattr(self, 'royale_lora_start_var'):
            data["royale_lora_start"] = self.royale_lora_start_var.get()
            data["royale_lora_end"] = self.royale_lora_end_var.get()
            data["royale_lora_frames"] = self.royale_lora_frames_var.get()
            data["royale_lora_w"] = self.royale_lora_w_var.get()
            data["royale_lora_h"] = self.royale_lora_h_var.get()
        if hasattr(self, 'royale_cmp_mode_var'):
            data["royale_cmp_mode"] = self.royale_cmp_mode_var.get()
            data["royale_cmp_trigger"] = self.royale_cmp_trigger_var.get()
            data["royale_cmp_seed"] = self.royale_cmp_seed_var.get()
            data["royale_cmp_w"] = self.royale_cmp_w_var.get()
            data["royale_cmp_h"] = self.royale_cmp_h_var.get()
            data["royale_cmp_epochs"] = self.royale_cmp_epochs_var.get()
            try:
                data["royale_cmp_prompts"] = self.royale_cmp_prompts.get("1.0", tk.END).strip()
            except Exception:
                pass
        if hasattr(self, 'royale_pt_prompt_var'):
            data["royale_pt_prompt"] = self.royale_pt_prompt_var.get()
            data["royale_pt_dim"] = self.royale_pt_dim_var.get()
            data["royale_pt_custom"] = self.royale_pt_custom_var.get()
            data["royale_pt_frames"] = self.royale_pt_frames_var.get()
            data["royale_pt_ref"] = self.royale_pt_ref_var.get()
            data["royale_pt_w"] = self.royale_pt_w_var.get()
            data["royale_pt_h"] = self.royale_pt_h_var.get()
            data["royale_pt_use_epoch_ref"] = bool(self.royale_pt_use_epoch_ref_var.get())
            data["royale_pt_ref_strength"] = self.royale_pt_ref_strength_var.get()
            data["royale_pt_vary_seed"] = bool(self.royale_pt_vary_seed_var.get())
            data["royale_pt_ref_mp"] = self.royale_pt_ref_mp_var.get()
            data["royale_pt_seq_ref"] = bool(self.royale_pt_seq_ref_var.get())
            data["royale_pt_anchor"] = bool(self.royale_pt_anchor_var.get())
            data["royale_pt_anchor_str"] = self.royale_pt_anchor_str_var.get()
            data["royale_pt_start"] = self.royale_pt_start_var.get()
            data["royale_pt_end"] = self.royale_pt_end_var.get()
            data["royale_pt_interp"] = self.royale_pt_interp_var.get()
            data["royale_pt_drift"] = self.royale_pt_drift_var.get()
            data["royale_pt_subject"] = self.royale_pt_subject_var.get()
        # Remember whether the bottom status bar is shown
        data["status_bar_visible"] = bool(getattr(self, "_status_bar_visible", True))
        save_last_used(data)

    def _save_pref(self, key):
        """Save a single pref value back to prefs.json."""
        if key in self.prefs_vars:
            self.prefs[key] = self.prefs_vars[key].get()
            save_prefs(self.prefs)

    # ------------------------------------------------------------------
    # Live VRAM / RAM status bar (bottom of window)
    # ------------------------------------------------------------------
    def _build_status_bar(self, master):
        """Bottom status panel: stacked VRAM + system-RAM gradient fill bars (with
        per-run peak ticks) on the left, the live sample override on the right,
        and a remembered hide/show toggle. A daemon thread does the reads so the
        Tk redraw never stalls on an nvidia-smi call."""
        container = tk.Frame(master, bg=COLORS["bg_deep"])
        container.pack(side=tk.BOTTOM, fill=tk.X)
        self._status_container = container

        # Thin always-visible handle carrying the show/hide toggle.
        handle = tk.Frame(container, bg=COLORS["bg_deep"])
        handle.pack(side=tk.BOTTOM, fill=tk.X)
        self._status_handle = handle
        self._status_toggle_btn = tk.Button(
            handle, text="▾ Hide stats", font=(FONT_FAMILY, 8),
            bg=COLORS["bg_deep"], fg=COLORS["text_muted"],
            activebackground=COLORS["bg_surface"], activeforeground=COLORS["text_primary"],
            relief="flat", bd=0, padx=10, pady=1, cursor="hand2",
            command=self._toggle_status_bar)
        self._status_toggle_btn.pack(side=tk.RIGHT, padx=(0, 12))

        # The expandable bar (sits above the handle).
        bar = tk.Frame(container, bg=COLORS["bg_deep"], height=82)
        bar.pack(side=tk.BOTTOM, fill=tk.X, before=handle)
        bar.pack_propagate(False)
        self._status_bar_frame = bar

        # --- left: stacked VRAM (top) + RAM (bottom) gradient bars ---
        bars_col = tk.Frame(bar, bg=COLORS["bg_deep"])
        bars_col.pack(side=tk.LEFT, padx=(14, 18), pady=10)
        self._vram_canvas = tk.Canvas(bars_col, width=360, height=27, bg=COLORS["bg_surface"],
                                      highlightthickness=0)
        self._vram_canvas.pack(side=tk.TOP, pady=(0, 6))
        self._ram_canvas = tk.Canvas(bars_col, width=360, height=27, bg=COLORS["bg_surface"],
                                     highlightthickness=0)
        self._ram_canvas.pack(side=tk.TOP)

        # --- far right: training-queue button (lower-right corner of the app) ---
        # Packed BEFORE the override panel so it owns the corner; the override panel's
        # expand soaks up whatever is left in the middle.
        # fill=Y on both this column and the override panel below is what makes the two
        # blocks exactly the same height (the bar is a fixed 82 px with pack_propagate off,
        # so each ends up 82 - 2*pady). Without it each block sizes to its own content and
        # the button sat visibly shorter than the panel beside it.
        qcol = tk.Frame(bar, bg=COLORS["bg_deep"])
        qcol.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 14), pady=10)
        self._queue_btn = tk.Button(
            qcol, text="📋 Queue", font=(FONT_FAMILY, 9, "bold"),
            bg=COLORS["queue_blue"], fg=COLORS["bg_deep"],
            activebackground=COLORS["queue_blue_hover"], activeforeground=COLORS["bg_deep"],
            relief="flat", bd=0, padx=12, cursor="hand2",
            command=self._open_queue_window)
        self._queue_btn.pack(fill=tk.BOTH, expand=True)
        self._refresh_queue_button()

        # --- right: live sample override (surface-coloured mini panel) ---
        # Widths trimmed vs the original layout to make room for the queue button.
        ov = tk.Frame(bar, bg=COLORS["bg_surface"])
        ov.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=10, ipadx=8, ipady=4)
        _sbg = COLORS["bg_surface"]
        r1 = tk.Frame(ov, bg=_sbg); r1.pack(fill=tk.X, padx=8, pady=(4, 0))
        self.sample_override_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(r1, text="Override next sample", variable=self.sample_override_var,
                        command=self._on_sample_override_changed,
                        style="Surface.TCheckbutton").pack(side=tk.LEFT)
        tk.Label(r1, text="seed", bg=_sbg, fg=COLORS["text_muted"],
                 font=(FONT_FAMILY, 8)).pack(side=tk.LEFT, padx=(10, 3))
        self.sample_override_seed_var = tk.StringVar(value="1234")
        ttk.Entry(r1, textvariable=self.sample_override_seed_var, width=6).pack(side=tk.LEFT)
        # Same list as the Samples tab (SAMPLE_RESOLUTIONS) — these two had drifted, and
        # the override's lower ceiling silently downgraded a 1280/1536 preview.
        _res_vals = SAMPLE_RESOLUTIONS
        tk.Label(r1, text="W", bg=_sbg, fg=COLORS["text_muted"],
                 font=(FONT_FAMILY, 8)).pack(side=tk.LEFT, padx=(8, 3))
        self.sample_override_w_var = tk.StringVar(value="768")
        ttk.Combobox(r1, textvariable=self.sample_override_w_var, values=_res_vals,
                     state="readonly", width=5).pack(side=tk.LEFT)
        tk.Label(r1, text="H", bg=_sbg, fg=COLORS["text_muted"],
                 font=(FONT_FAMILY, 8)).pack(side=tk.LEFT, padx=(8, 3))
        self.sample_override_h_var = tk.StringVar(value="768")
        ttk.Combobox(r1, textvariable=self.sample_override_h_var, values=_res_vals,
                     state="readonly", width=5).pack(side=tk.LEFT)
        # Reference image — auto-capped to ~0.20 MP by the trainer so a big image can't OOM the
        # sample. Shown for BOTH families: Klein uses it as edit conditioning, Krea 2 routes it
        # through the Qwen3-VL vision path. (This comment used to claim it was hidden under
        # Krea 2 — it never was; there is no hide call for these widgets anywhere.)
        self._override_ref_caption = tk.Label(r1, text="Ref", bg=_sbg, fg=COLORS["text_muted"],
                 font=(FONT_FAMILY, 8))
        self._override_ref_caption.pack(side=tk.LEFT, padx=(8, 3))
        self.sample_override_ref_var = tk.StringVar(value="")
        # Compact button so it matches the seed/resolution input height (the
        # default ttk.Button padding is taller and pushes the prompt row down).
        ttk.Style().configure("OverrideRef.TButton", padding=(8, 0), font=(FONT_FAMILY, 8))
        self._override_ref_browse_btn = ttk.Button(r1, text="Browse…", width=9, style="OverrideRef.TButton",
                   command=self._browse_override_ref)
        self._override_ref_browse_btn.pack(side=tk.LEFT)
        self._override_ref_label = tk.Label(r1, text="(none)", bg=_sbg,
                                            fg=COLORS["text_muted"], font=(FONT_FAMILY, 8))
        self._override_ref_label.pack(side=tk.LEFT, padx=(6, 2))
        self._override_ref_clear_btn = tk.Button(r1, text="✕", font=(FONT_FAMILY, 8), bg=_sbg, fg=COLORS["text_muted"],
                  activebackground=COLORS["border"], activeforeground=COLORS["text_primary"],
                  relief="flat", bd=0, cursor="hand2",
                  command=self._clear_override_ref)
        self._override_ref_clear_btn.pack(side=tk.LEFT)
        r2 = tk.Frame(ov, bg=_sbg); r2.pack(fill=tk.X, padx=8, pady=(8, 4))
        tk.Label(r2, text="Prompt", bg=_sbg, fg=COLORS["text_muted"],
                 font=(FONT_FAMILY, 8)).pack(side=tk.LEFT, padx=(0, 6))
        self.sample_override_prompt_var = tk.StringVar()
        ttk.Entry(r2, textvariable=self.sample_override_prompt_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        for _v in (self.sample_override_prompt_var, self.sample_override_seed_var,
                   self.sample_override_w_var, self.sample_override_h_var,
                   self.sample_override_ref_var):
            _v.trace_add("write", lambda *a: self._on_sample_override_changed())

        self._vram_peak = 0
        self._ram_peak = 0
        self._status_latest = (None, None)
        self._status_stop = False
        # Restore remembered visibility (default shown).
        self._status_bar_visible = bool(self.last_used.get("status_bar_visible", True))
        if not self._status_bar_visible:
            bar.pack_forget()
            self._status_toggle_btn.configure(text="▴ Show stats")
        import threading
        self._status_thread = threading.Thread(target=self._status_reader_loop, daemon=True)
        self._status_thread.start()
        self.master.after(800, self._poll_status_bar)

    def _toggle_status_bar(self):
        """Show/hide the stats bar; remember the choice across launches."""
        self._status_bar_visible = not getattr(self, "_status_bar_visible", True)
        if self._status_bar_visible:
            self._status_bar_frame.pack(side=tk.BOTTOM, fill=tk.X, before=self._status_handle)
            self._status_toggle_btn.configure(text="▾ Hide stats")
        else:
            self._status_bar_frame.pack_forget()
            self._status_toggle_btn.configure(text="▴ Show stats")
        try:
            self._save_last_used_paths()
        except Exception:
            pass

    def _visible_gpu_index(self):
        """The physical GPU index training will actually use.

        NVML and nvidia-smi index every card in the machine; CUDA_VISIBLE_DEVICES does not
        change that, it only changes what torch can see. So on a two-GPU box with
        CUDA_VISIBLE_DEVICES=1, training runs on physical card 1 while an unqualified NVML read
        reports card 0 — the status bar then shows a card that is doing nothing (issue #60).
        Torch's cuda:0 is the FIRST entry in the list, hence [0]. When CUDA_VISIBLE_DEVICES
        holds a UUID (new format), map it back to NVML index via _gpu_info."""
        raw = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
        if raw.isdigit():
            return int(raw)
        # UUID — look up NVML index from _gpu_info
        for _k, _info in getattr(self, "_gpu_info", {}).items():
            if _info[3] == raw:
                return _info[0]
        return 0

    def _read_vram(self):
        """Return (used_bytes, total_bytes) for the GPU training uses, or None. Prefers pynvml
        (fast); falls back to a one-shot nvidia-smi query. AMD ROCm paths are
        tried only when NVIDIA readers return nothing."""
        try:
            import pynvml
            if not getattr(self, "_nvml_init", False):
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self._visible_gpu_index())
                self._nvml_init = True
            m = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            return int(m.used), int(m.total)
        except Exception:
            pass
        try:
            import subprocess
            out = subprocess.run(
                ["nvidia-smi", "-i", str(self._visible_gpu_index()),
                 "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            used, total = out.stdout.strip().splitlines()[0].split(",")
            return int(used) * 1024 * 1024, int(total) * 1024 * 1024
        except Exception:
            pass
        try:
            from fizgig.utils.vram_monitor import read_amd_gpu_vram
            return read_amd_gpu_vram()
        except Exception:
            return None

    def _status_reader_loop(self):
        import time
        while not getattr(self, "_status_stop", False):
            vram = self._read_vram()
            try:
                import psutil
                vm = psutil.virtual_memory()
                ram = (vm.total - vm.available, vm.total)
            except Exception:
                ram = None
            self._status_latest = (vram, ram)
            time.sleep(1.0)

    @staticmethod
    def _lerp_color(c1, c2, t):
        t = max(0.0, min(1.0, t))
        r = round(int(c1[1:3], 16) + (int(c2[1:3], 16) - int(c1[1:3], 16)) * t)
        g = round(int(c1[3:5], 16) + (int(c2[3:5], 16) - int(c1[3:5], 16)) * t)
        b = round(int(c1[5:7], 16) + (int(c2[5:7], 16) - int(c1[5:7], 16)) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _draw_status_segment(self, canvas, used, total, peak, label, c_start, c_end):
        canvas.delete("all")
        try:
            w = int(canvas["width"]); h = int(canvas["height"])
        except Exception:
            return
        frac = max(0.0, min(1.0, used / total)) if total else 0.0
        # track
        canvas.create_rectangle(0, 0, w, h, fill=COLORS["bg_deep"], outline="")
        # gradient fill: colour interpolates c_start -> c_end across the FULL
        # width, drawn up to the current fill (fuller = closer to c_end).
        fill_w = int(w * frac)
        step = 3
        for x in range(0, fill_w, step):
            col = self._lerp_color(c_start, c_end, x / max(1, w - 1))
            canvas.create_rectangle(x, 1, min(x + step, fill_w), h - 1, fill=col, outline="")
        # per-run peak tick
        if peak and total:
            px = int(w * max(0.0, min(1.0, peak / total)))
            canvas.create_line(px, 0, px, h, fill="#FFFFFF", width=2)
        canvas.create_text(10, h // 2,
                           text=(f"{label}  {used/1073741824:.1f} / {total/1073741824:.1f} GB"
                                 f" · peak {peak/1073741824:.1f}"),  # GiB (binary) — matches '32 GB' labels
                           anchor="w", fill="#FFFFFF", font=(FONT_FAMILY, 9, "bold"))

    def _poll_status_bar(self):
        vram, ram = getattr(self, "_status_latest", (None, None))
        visible = getattr(self, "_status_bar_visible", True)
        if vram:
            u, t = vram
            self._vram_peak = max(self._vram_peak, u)
            if visible:
                self._draw_status_segment(self._vram_canvas, u, t, self._vram_peak,
                                          "VRAM", "#3FB950", "#E5534B")  # green → red
        if ram:
            u, t = ram
            self._ram_peak = max(self._ram_peak, u)
            if visible:
                self._draw_status_segment(self._ram_canvas, u, t, self._ram_peak,
                                          "RAM", "#3B82F6", "#EAC54F")   # blue → yellow
        self.master.after(1000, self._poll_status_bar)

    def reset_status_peaks(self):
        """Zero the VRAM/RAM peak markers — call at the start of a training run."""
        self._vram_peak = 0
        self._ram_peak = 0

    def _sample_override_path(self):
        # Prefer the live entry (always current) so this matches the --output_dir
        # the trainer is launched with, even before settings is synced.
        out_dir = ""
        try:
            out_dir = self.entries["LORA_OUTPUT_DIR"].get().strip()
        except Exception:
            pass
        if not out_dir:
            out_dir = self.settings.get("LORA_OUTPUT_DIR", "") or "."
        return os.path.join(out_dir, ".sample_override.json")

    def _on_sample_override_changed(self):
        """Write or remove the live sample-override sentinel the trainer reads.

        Active when the toggle is on AND there's a prompt OR a reference image. A reference with
        no prompt is a valid Krea 2 'generate from this picture' override (routed through the
        Qwen3-VL vision path); the Klein trainer still ignores prompt-less overrides (it requires
        a prompt). Off removes the sentinel → samples fall back to the Samples tab."""
        path = self._sample_override_path()
        try:
            active = self.sample_override_var.get() and (
                self.sample_override_prompt_var.get().strip()
                or self.sample_override_ref_var.get().strip())
            if active:
                try:
                    seed = int(self.sample_override_seed_var.get() or "1234")
                except ValueError:
                    seed = 1234
                try:
                    width = int(self.sample_override_w_var.get() or "768")
                except ValueError:
                    width = 768
                try:
                    height = int(self.sample_override_h_var.get() or "768")
                except ValueError:
                    height = 768
                data = {"prompt": self.sample_override_prompt_var.get().strip(),
                        "seed": seed, "width": width, "height": height,
                        "ref_image": self.sample_override_ref_var.get().strip()}
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                # Atomic: the trainer polls this file at epoch boundaries, and a plain write
                # can be caught half-finished mid-keystroke — unparseable JSON reads as "no
                # override" for that epoch.
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                os.replace(tmp, path)
            elif os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    def _browse_override_ref(self):
        """Pick a reference image for the live sample override (Klein edit
        conditioning). The trainer auto-caps it to ~0.20 MP, so any size is safe."""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select reference image for samples",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")],
            initialdir=self._pref_initialdir("input_ref_dir"))
        if path:
            self.sample_override_ref_var.set(path)
            self._override_ref_label.configure(text=os.path.basename(path)[:24])
            self._on_sample_override_changed()

    def _clear_override_ref(self):
        self.sample_override_ref_var.set("")
        self._override_ref_label.configure(text="(none)")
        self._on_sample_override_changed()

    def _browse_sample_ref(self):
        """Pick the persistent reference image for training samples (Samples tab)."""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select reference image for samples",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")],
            initialdir=self._pref_initialdir("input_ref_dir"))
        if path:
            self.sample_ref_image_var.set(path)