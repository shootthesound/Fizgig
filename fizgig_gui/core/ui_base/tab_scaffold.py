import json
import os
import threading

import tkinter as tk
from tkinter import ttk

from fizgig_gui.core.config.constants import COLORS, FONT_FAMILY
from fizgig_gui.core.config.prefs import HELP_FILE, _check_for_update, _git_describe_version, _running_on_pod
from fizgig_gui.core.config.settings_map import SETTING_TO_PREF
from fizgig_gui.core.domain.architectures import ARCHITECTURES


class TabScaffoldMixin:
    def _add_tab_banner(self, parent, title, subtitle):
        """Start-tab-style banner (22pt title + 11pt subtitle on bg_deep).
        Packs into `parent`; returns the banner frame in case the caller wants to tweak it."""
        banner = tk.Frame(parent, bg=COLORS["bg_deep"])
        banner.pack(fill=tk.X, padx=36, pady=(28, 20))
        tk.Label(banner, text=title,
                 font=(FONT_FAMILY, 22, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_deep"]).pack(anchor=tk.W)
        if subtitle:
            # Colour only — 11pt is already comfortable, and this passed contrast before. Moved
            # onto text_explain so every piece of explanatory copy is one colour, rather than the
            # subtitle being a shade apart from the card descriptions directly beneath it.
            tk.Label(banner, text=subtitle,
                     font=(FONT_FAMILY, 11),
                     fg=COLORS["text_explain"], bg=COLORS["bg_deep"],
                     wraplength=1050, justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))
        return banner

    def _add_youtube_help_button(self, parent, tab_key="start", prominent=False):
        """Add a 'Tutorial' button at the bottom of a tab's outer frame.

        Labelled just "Tutorial" rather than "Get help on YouTube": the play glyph and the red
        already say where it goes, and "get help" reads as troubleshooting when these are walk-
        throughs. Shorter also keeps the Start tab's button row on one line.

        `tab_key` selects the URL from help.json's `youtube_urls` dict.
        `prominent=True` uses a larger button and the Start tab's extra buttons alongside.
        """
        # Used only when help.json is missing or unreadable — point it at the real guide
        # rather than the old joke URL, since that path fires on a genuine error.
        fallback = "https://www.youtube.com/watch?v=yrz0l6URGGk"
        try:
            with open(HELP_FILE, "r", encoding="utf-8") as f:
                urls = json.load(f).get("youtube_urls", {})
            url = urls.get(tab_key, fallback)
        except Exception:
            url = fallback
        btn_frame = tk.Frame(parent, bg=COLORS["bg_deep"])
        btn_frame.pack(fill=tk.X, padx=36, pady=(16, 16) if prominent else (8, 8))
        row = tk.Frame(btn_frame, bg=COLORS["bg_deep"])
        row.pack(anchor=tk.W)
        btn = tk.Button(
            row, text="\u25b6  Tutorial",
            font=(FONT_FAMILY, 12 if prominent else 10, "bold"),
            fg="#FFFFFF", bg="#CC0000", activeforeground="#FFFFFF", activebackground="#990000",
            relief="flat", bd=0, padx=20 if prominent else 16,
            pady=6, cursor="hand2",
            command=lambda: __import__("webbrowser").open(url),
        )
        btn.pack(side=tk.LEFT)
        if prominent:
            coffee = tk.Button(
                row, text="\u2615  Buy me a coffee",
                font=(FONT_FAMILY, 12, "bold"),
                fg="#000000", bg="#FFDD00", activeforeground="#000000", activebackground="#E5C700",
                relief="flat", bd=0, padx=20, pady=6, cursor="hand2",
                command=lambda: __import__("webbrowser").open(
                    "https://buymeacoffee.com/lorasandlenses"),
            )
            coffee.pack(side=tk.LEFT, padx=(12, 0))
            # Shown on a pod too, deliberately: the Preferences card already tells a pod user they
            # are on one, and a user who is happy renting is exactly who might send the link on.
            # RunPod's own violet, so it reads as theirs rather than as another Fizgig action.
            runpod = tk.Button(
                row, text="⚡  Deploy on RunPod",
                font=(FONT_FAMILY, 12, "bold"),
                fg="#FFFFFF", bg="#7C3AED", activeforeground="#FFFFFF", activebackground="#6D28D9",
                relief="flat", bd=0, padx=20, pady=6, cursor="hand2",
                command=lambda: __import__("webbrowser").open(self._runpod_deploy_url()),
            )
            runpod.pack(side=tk.LEFT, padx=(12, 0))
            # width sized for "⬆ Update Available" so the label can flip to it after the
            # startup update check without shifting the whole button row.
            about = tk.Button(
                row, text="About", width=16,
                font=(FONT_FAMILY, 12, "bold"),
                fg="#FFFFFF", bg=COLORS["accent"],
                activeforeground="#FFFFFF", activebackground=COLORS["accent_hover"],
                relief="flat", bd=0, padx=20, pady=6, cursor="hand2",
                command=self._open_about_dialog,
            )
            about.pack(side=tk.LEFT, padx=(12, 0))
            self._about_btn = about

    # ------------------------------------------------------------
    # Update check (startup + manual from the About dialog)
    # ------------------------------------------------------------

    def _start_update_check(self, on_done=None):
        """Run the git-based release check on a daemon thread; marshal the result back to
        the Tk thread. Silent by design: 'unknown' (offline, no origin) shows nothing —
        never a false nag, never an error popup."""
        def _worker():
            status, info = _check_for_update()
            self.master.after(0, lambda: self._apply_update_status(status, info, on_done))
        threading.Thread(target=_worker, daemon=True).start()

    def _apply_update_status(self, status, info, on_done=None):
        if status == "update_available":
            self._update_info = info                     # the newer tag, e.g. "v3.1.2"
            btn = getattr(self, "_about_btn", None)
            if btn is not None:
                try:
                    btn.config(text="⬆ Update Available",
                               bg=COLORS["warning"], activebackground="#D97706",
                               fg="#1E2530", activeforeground="#1E2530")
                except Exception:
                    pass
        elif status == "up_to_date":
            self._update_info = None
            btn = getattr(self, "_about_btn", None)
            if btn is not None:
                try:
                    btn.config(text="About", bg=COLORS["accent"],
                               activebackground=COLORS["accent_hover"],
                               fg="#FFFFFF", activeforeground="#FFFFFF")
                except Exception:
                    pass
        if on_done:
            try:
                on_done(status, info)
            except Exception:
                pass

    def _open_about_dialog(self):
        """A small, personal About window: who made Fizgig, why, and a no-pressure
        nudge to the tip jar."""
        import webbrowser
        win = tk.Toplevel(self.master)
        win.title("About Fizgig")
        win.configure(bg=COLORS["bg_deep"])
        win.transient(self.master)
        win.resizable(False, False)
        try:
            win.grab_set()
        except Exception:
            pass

        pad = tk.Frame(win, bg=COLORS["bg_deep"])
        pad.pack(fill=tk.BOTH, expand=True, padx=28, pady=24)
        WRAP = 460

        def heading(text, size=20, fg=None, pady=(0, 2)):
            tk.Label(pad, text=text, font=(FONT_FAMILY, size, "bold"),
                     fg=fg or COLORS["text_primary"], bg=COLORS["bg_deep"]).pack(anchor=tk.W, pady=pady)

        def para(text, fg=None, italic=False, pady=(0, 10)):
            tk.Label(pad, text=text, font=(FONT_FAMILY, 10, "italic" if italic else "normal"),
                     fg=fg or COLORS["text_secondary"], bg=COLORS["bg_deep"],
                     wraplength=WRAP, justify=tk.LEFT).pack(anchor=tk.W, pady=pady)

        def link(text, url, pady=(0, 2)):
            lbl = tk.Label(pad, text=text, font=(FONT_FAMILY, 10, "underline"),
                           fg=COLORS["accent_hover"], bg=COLORS["bg_deep"], cursor="hand2")
            lbl.pack(anchor=tk.W, pady=pady)
            lbl.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))

        heading("Fizgig", 22)
        tk.Label(pad, text="Klein 9B & Krea 2 LoRA Studio — by Peter Neill",
                 font=(FONT_FAMILY, 11), fg=COLORS["text_explain"],
                 bg=COLORS["bg_deep"]).pack(anchor=tk.W, pady=(0, 4))
        tk.Label(pad, text=f"Version {_git_describe_version() or 'unknown'}",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"],
                 bg=COLORS["bg_deep"]).pack(anchor=tk.W, pady=(0, 2))
        # Always shown — how THIS install updates, whether or not one is pending right now.
        _how = ("On RunPod, Fizgig updates itself every time you restart the pod — just stop "
                "and start it to get the latest."
                if _running_on_pod() else
                ("To update: close Fizgig and run update_fizgig_rocm.bat."
                 if os.environ.get("FIZGIG_GPU_BACKEND", "").lower() == "rocm" else
                 "To update: close Fizgig and run update_fizgig.bat."))
        tk.Label(pad, text=_how, font=(FONT_FAMILY, 9), fg=COLORS["text_muted"],
                 bg=COLORS["bg_deep"], wraplength=WRAP, justify=tk.LEFT).pack(
            anchor=tk.W, pady=(0, 10))

        # Update banner — its own frame so the "Check for updates" button below can rebuild it
        # live without reopening the dialog. Rendered per the current self._update_info.
        update_box = tk.Frame(pad, bg=COLORS["bg_deep"])
        update_box.pack(fill=tk.X, pady=(0, 8))

        def _render_update_box():
            for w in update_box.winfo_children():
                w.destroy()
            tag = getattr(self, "_update_info", None)
            if not tag:
                return
            card = tk.Frame(update_box, bg=COLORS["accent_subtle"],
                            highlightbackground=COLORS["warning"], highlightthickness=1)
            card.pack(fill=tk.X)
            tk.Label(card, text=f"⬆  Fizgig {tag} is available",
                     font=(FONT_FAMILY, 11, "bold"), fg=COLORS["text_primary"],
                     bg=COLORS["accent_subtle"]).pack(anchor=tk.W, padx=12, pady=(10, 0))
            _upd = ("Restart the pod (stop and start it) to update."
                    if _running_on_pod() else
                    ("Close Fizgig and run update_fizgig_rocm.bat to update."
                     if os.environ.get("FIZGIG_GPU_BACKEND", "").lower() == "rocm" else
                     "Close Fizgig and run update_fizgig.bat to update."))
            tk.Label(card, text=f"You're on {_git_describe_version() or 'an older build'}. {_upd}",
                     font=(FONT_FAMILY, 9), fg=COLORS["text_explain"], bg=COLORS["accent_subtle"],
                     wraplength=WRAP - 24, justify=tk.LEFT).pack(anchor=tk.W, padx=12, pady=(2, 8))
            notes = tk.Label(card, text="📖  Read the release notes",
                             font=(FONT_FAMILY, 10, "underline"), fg=COLORS["accent_hover"],
                             bg=COLORS["accent_subtle"], cursor="hand2")
            notes.pack(anchor=tk.W, padx=12, pady=(0, 10))
            notes.bind("<Button-1>", lambda e: webbrowser.open(
                "https://github.com/shootthesound/Fizgig/releases/latest"))

        _render_update_box()

        para("By trade I'm a photographer and videographer — mostly live music, portraits, and a "
             "bit of teaching — and an AI tinkerer by night. I build a lot of open-source tooling "
             "for ComfyUI and Klein/Flux workflows.")
        link("Photography — shootthesound.com", "https://shootthesound.com")
        link("Code & ComfyUI nodes — github.com/shootthesound", "https://github.com/shootthesound", pady=(0, 4))
        para("(Realtime-LoRA, LongLook, Angelo, mesh and a couple of dozen more — Fizgig grew out of that world.)",
             fg=COLORS["text_muted"], pady=(0, 16))

        tk.Frame(pad, bg=COLORS["border"], height=1).pack(fill=tk.X, pady=(0, 16))

        para("A quiet note: a lot of this got built in the small hours. The last year has been a hard "
             "one for our family — one of my children has been facing some serious health challenges — "
             "and honestly, losing myself in making and obsessing over tools like this is how I carve out "
             "a little headspace. It keeps my hands busy and my head somewhere steady.", italic=True)

        para("Fizgig is free and always will be. If it's useful to you and you'd like to drop a coffee in "
             "the tip jar, it genuinely means a lot right now — but it's in no way an obligation. Using it "
             "and enjoying it is more than enough. Thank you for being here.",
             fg=COLORS["text_secondary"], pady=(0, 18))

        btn_row = tk.Frame(pad, bg=COLORS["bg_deep"])
        btn_row.pack(anchor=tk.W)
        # No coffee button here on purpose — it's already on the Start tab, so
        # repeating it in the popup would feel pushy. The note above is enough.
        tk.Button(btn_row, text="Close", font=(FONT_FAMILY, 11),
                  fg=COLORS["text_primary"], bg=COLORS["bg_surface"],
                  activeforeground=COLORS["text_primary"], activebackground=COLORS["border"],
                  relief="flat", bd=0, padx=18, pady=8, cursor="hand2",
                  command=win.destroy).pack(side=tk.LEFT)

        check_btn = tk.Button(btn_row, text="Check for updates", font=(FONT_FAMILY, 11),
                              fg=COLORS["text_primary"], bg=COLORS["bg_surface"],
                              activeforeground=COLORS["text_primary"],
                              activebackground=COLORS["border"],
                              relief="flat", bd=0, padx=18, pady=8, cursor="hand2")
        check_btn.pack(side=tk.LEFT, padx=(12, 0))

        def _manual_check():
            if not win.winfo_exists():
                return
            check_btn.config(text="Checking…", state=tk.DISABLED)

            def _done(status, info):
                if not win.winfo_exists():
                    return
                _render_update_box()
                check_btn.config(
                    state=tk.NORMAL,
                    text=("Up to date ✓" if status == "up_to_date"
                          else "Couldn't check" if status == "unknown"
                          else "Check for updates"))
            self._start_update_check(on_done=_done)

        check_btn.config(command=_manual_check)

        win.update_idletasks()
        # Centre over the main window.
        try:
            px = self.master.winfo_rootx() + (self.master.winfo_width() - win.winfo_width()) // 2
            py = self.master.winfo_rooty() + (self.master.winfo_height() - win.winfo_height()) // 3
            win.geometry(f"+{max(0, px)}+{max(0, py)}")
        except Exception:
            pass

    def _start_section_card(self, parent, title, description=None, accent_border=False):
        """Start-tab-style surface card with an optional description line.
        Returns the inner content frame (caller packs/grids its widgets into it).
        The card is packed into `parent` with horizontal padding matching the banner."""
        outer = tk.Frame(parent, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.X, padx=36, pady=(0, 16))

        card = tk.Frame(outer, bg=COLORS["bg_surface"],
                        highlightbackground=COLORS["accent"] if accent_border else COLORS["border"],
                        highlightthickness=1, bd=0)
        card.pack(fill=tk.X)

        if title:
            tk.Label(card, text=title,
                     font=(FONT_FAMILY, 12, "bold"),
                     fg=COLORS["text_primary"], bg=COLORS["bg_surface"]).pack(
                anchor=tk.W, padx=20, pady=(16, 2 if description else 10)
            )
        desc_label = None
        if description:
            desc_label = tk.Label(card, text=description,
                                  font=(FONT_FAMILY, 10),
                                  fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                                  wraplength=760, justify=tk.LEFT)
            desc_label.pack(anchor=tk.W, padx=20, pady=(0, 10))

        content = tk.Frame(card, bg=COLORS["bg_surface"])
        content.pack(fill=tk.X, padx=20, pady=(0, 16))
        # Stashed so a caller can reword the description later (the Samples tab retitles its
        # cards per model family). None when the card was built without one.
        content._desc_label = desc_label
        return content

    def _add_field_to_section(self, parent, key, label_text, input_type, row):
        """Helper method to add a field to a section (collapsible or regular frame)"""
        # Create label
        label = tk.Label(
            parent,
            text=f"{label_text}:",
            font=(FONT_FAMILY, 10),
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_surface"]
        )
        label.grid(row=row, column=0, sticky=tk.W, padx=(12, 8), pady=4)
        self.labels[key] = label

        # Create entry/combobox based on type
        if input_type == "dropdown":
            if key == "MODEL_TYPE":
                arch = self.settings["ARCHITECTURE"]
                arch_config = ARCHITECTURES.get(arch, ARCHITECTURES["Flux 2 Klein Base 9B"])
                model_types = arch_config.get("model_types", ["t2v-14B", "i2v-14B"])
                var = tk.StringVar(value=self.settings[key])
                self.entries[key] = ttk.Combobox(parent, textvariable=var, values=model_types, state="readonly", width=38)
                if self.settings[key] in model_types:
                    self.entries[key].current(model_types.index(self.settings[key]))
                else:
                    self.entries[key].current(0)
            elif key == "OPTIMIZER_TYPE":
                var = tk.StringVar(value=self.settings[key])
                self.entries[key] = ttk.Combobox(parent, textvariable=var, values=self.optimizer_types, state="readonly", width=38)
                # Saved settings may carry a name that's no longer offered (e.g. the removed
                # adafactor/prodigy/came) — fall back to the default instead of crashing.
                if self.settings[key] in self.optimizer_types:
                    self.entries[key].current(self.optimizer_types.index(self.settings[key]))
                else:
                    self.entries[key].set("adamw8bit")
            elif key == "LR_SCHEDULER":
                lr_scheduler_options = ["constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear", "polynomial"]
                self.lr_scheduler_var = tk.StringVar(value=self.settings["LR_SCHEDULER"])
                self.entries[key] = ttk.Combobox(parent, textvariable=self.lr_scheduler_var, values=lr_scheduler_options, state="readonly", width=38)
        else:
            # Check if this entry should be bound to a shared pref var
            pref_key = SETTING_TO_PREF.get(key)
            if pref_key and pref_key in self.prefs_vars:
                self.entries[key] = ttk.Entry(parent, width=40, textvariable=self.prefs_vars[pref_key])
            else:
                self.entries[key] = ttk.Entry(parent, width=40)
                self.entries[key].insert(0, str(self.settings.get(key, "")))

        self.entries[key].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=4)

        # Create browse button if needed
        browse_btn = None
        if input_type in ["file", "directory"]:
            browse_btn = ttk.Button(parent, text="Browse", command=lambda k=key, t=input_type: self.browse_file(k, t))
            browse_btn.grid(row=row, column=2, sticky=tk.W, padx=(5, 12), pady=4)

        # Store row info for show/hide functionality
        self.rows[key] = {"row": row, "label": label, "entry": self.entries[key], "browse": browse_btn, "parent": parent}
