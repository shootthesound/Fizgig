"""Fizgig launch splash — the window you see while the real one is being built.

On a slow drive the first launch can take minutes to show anything: the GUI module pulls
in torch, insightface and transformers before a single widget exists, and then thirteen
tabs get built. During all of that the user saw nothing and assumed the app had hung
(#115, fm3at; confirmed on NVMe too by Davikar). This puts a small window up first.

Deliberately tiny and dependency-free: it must appear BEFORE lora_trainer_gui.py is even
compiled, so it cannot import anything from it — the palette below is a copy of the
matching COLORS entries there (the one place hardcoded hex is unavoidable). It attaches
to the app's own Tk root (created withdrawn by the launcher), so there is exactly one Tk
interpreter and every image/variable the GUI later creates lands on the right root.
"""
import os
import tkinter as tk
from tkinter import ttk

HERE = os.path.dirname(os.path.abspath(__file__))

# Mirrors lora_trainer_gui.COLORS / FONT_FAMILY — keep in step by hand.
_BG_DEEP = "#1E2530"
_BG_SURFACE = "#252D38"
_TEXT_PRIMARY = "#F0F4F8"
_TEXT_SECONDARY = "#8A9BAE"
_ACCENT = "#3B82F6"
_FONT = "Segoe UI"


class Splash:
    """A borderless, centred status window on the app root. Call status() as phases
    complete (each call pumps the event loop so the window repaints) and close() once
    the main window is on screen. Every method swallows Tk errors: a splash must never
    be the reason the app fails to start."""

    def __init__(self, root, status="Starting…"):
        self.root = root
        self.top = None
        self._icon = None
        try:
            top = tk.Toplevel(root, bg=_BG_DEEP)
            top.overrideredirect(True)
            top.attributes("-topmost", True)
            w, h = 460, 176
            sw, sh = top.winfo_screenwidth(), top.winfo_screenheight()
            top.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
            border = tk.Frame(top, bg=_BG_SURFACE, padx=1, pady=1)
            border.pack(fill=tk.BOTH, expand=True)
            body = tk.Frame(border, bg=_BG_DEEP, padx=24, pady=18)
            body.pack(fill=tk.BOTH, expand=True)

            head = tk.Frame(body, bg=_BG_DEEP)
            head.pack(fill=tk.X)
            icon_path = os.path.join(HERE, "icon.png")
            if os.path.exists(icon_path):
                try:
                    img = tk.PhotoImage(file=icon_path)      # Tk reads PNG natively
                    f = max(1, img.width() // 40)
                    self._icon = img.subsample(f, f) if f > 1 else img
                    tk.Label(head, image=self._icon, bg=_BG_DEEP).pack(side=tk.LEFT,
                                                                       padx=(0, 12))
                except tk.TclError:
                    self._icon = None
            tk.Label(head, text="Fizgig", font=(_FONT, 20, "bold"), fg=_TEXT_PRIMARY,
                     bg=_BG_DEEP).pack(side=tk.LEFT)
            tk.Label(head, text="LoRA trainer & workbench", font=(_FONT, 10),
                     fg=_TEXT_SECONDARY, bg=_BG_DEEP).pack(side=tk.LEFT, padx=(10, 0),
                                                          pady=(6, 0))

            self._status = tk.StringVar(master=root, value=status)
            tk.Label(body, textvariable=self._status, font=(_FONT, 10),
                     fg=_TEXT_SECONDARY, bg=_BG_DEEP, anchor="w").pack(fill=tk.X,
                                                                       pady=(18, 8))
            style = ttk.Style(root)
            try:
                style.theme_use("clam")      # the only stock theme that honours colours here
            except tk.TclError:
                pass
            style.configure("Splash.Horizontal.TProgressbar", troughcolor=_BG_SURFACE,
                            background=_ACCENT, bordercolor=_BG_DEEP,
                            lightcolor=_ACCENT, darkcolor=_ACCENT)
            self._bar = ttk.Progressbar(body, mode="indeterminate", length=w - 50,
                                        style="Splash.Horizontal.TProgressbar")
            self._bar.pack(fill=tk.X)
            self._bar.start(12)
            self.top = top
            self.pump()
        except Exception:
            self.top = None

    def status(self, text):
        """Update the line under the title and repaint. Idle tasks only — a full update()
        here would fire the GUI constructor's own after() timers before the tabs they
        expect exist."""
        if self.top is None:
            return
        try:
            self._status.set(text)
            self._bar.step(6)          # visible motion even though no event loop is running
            self.top.update_idletasks()
        except Exception:
            pass

    def pump(self):
        """Full event pump — for the launcher's wait loop while the GUI module imports on a
        worker thread (nothing else exists yet, so there are no timers to fire early)."""
        if self.top is None:
            return
        try:
            self.top.update()
        except Exception:
            pass

    def close(self):
        if self.top is None:
            return
        try:
            self._bar.stop()
        except Exception:
            pass
        try:
            self.top.destroy()
        except Exception:
            pass
        self.top = None
