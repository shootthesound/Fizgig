import tkinter as tk

from fizgig_gui.core.config.constants import COLORS, FONT_FAMILY


class _GUIWriter:
    """Redirect stdout/stderr to the GUI log buffer when running under pythonw."""
    def __init__(self, gui_ref, stream_name):
        self._gui = gui_ref
        self._name = stream_name
    def write(self, text):
        if text and text.strip():
            try:
                self._gui.master.after(0, self._gui._append_global_log,
                                       f"[{self._name}] {text}")
            except Exception:
                pass
    def flush(self):
        pass


class ToolTip:
    """Simple tooltip class for tkinter widgets"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip_window:
            return
        bbox = self.widget.bbox("insert") if hasattr(self.widget, 'bbox') else None
        bx, by = (bbox[0], bbox[1]) if bbox else (0, 0)
        x = bx + self.widget.winfo_rootx() + 25
        y = by + self.widget.winfo_rooty() + 25

        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                        background=COLORS["bg_surface"], foreground=COLORS["text_primary"],
                        relief=tk.SOLID, borderwidth=1,
                        font=(FONT_FAMILY, 9), padx=8, pady=6)
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class CollapsibleFrame(tk.Frame):
    """
    A frame that can be collapsed/expanded with a header.

    Features:
    - Click header to toggle
    - Arrow indicator (▶/▼)
    - Optional badge showing field status
    - Maintains child widget state when collapsed
    - All child widgets remain accessible via parent.entries[]

    Styled as a Start-tab-style card (bg_surface body, bordered outer frame).
    """

    def __init__(self, parent, title, default_expanded=True, badge_callback=None):
        """
        Args:
            parent: Parent widget
            title: Section title text
            default_expanded: Whether section starts expanded
            badge_callback: Optional function returning (filled, total) tuple for badge
        """
        super().__init__(
            parent,
            bg=COLORS["bg_surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
            bd=0,
        )
        self.expanded = default_expanded
        self.title = title
        self.badge_callback = badge_callback

        # Create header frame
        self.header = tk.Frame(
            self,
            bg=COLORS["bg_header"],
            cursor="hand2"
        )
        self.header.pack(fill=tk.X)

        # Arrow indicator
        self.arrow = tk.Label(
            self.header,
            text="▼" if self.expanded else "▶",
            font=(FONT_FAMILY, 10),
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_header"],
            cursor="hand2"
        )
        self.arrow.pack(side=tk.LEFT, padx=(16, 10), pady=12)

        # Title label — matches Start-tab card headers at 12pt bold
        self.title_label = tk.Label(
            self.header,
            text=title,
            font=(FONT_FAMILY, 12, "bold"),
            fg=COLORS["text_primary"],
            bg=COLORS["bg_header"],
            cursor="hand2"
        )
        self.title_label.pack(side=tk.LEFT, pady=12)

        # Badge label (shows filled/total fields)
        self.badge = tk.Label(
            self.header,
            text="",
            font=(FONT_FAMILY, 9),
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_header"],
            cursor="hand2"
        )
        self.badge.pack(side=tk.RIGHT, padx=(8, 16), pady=12)

        # Content frame — bg_surface, padded from the card edge so children don't
        # touch the border. Children grid into this directly.
        self.content = tk.Frame(self, bg=COLORS["bg_surface"])
        if self.expanded:
            self.content.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 12))

        # Bind click events to all header elements
        for widget in [self.header, self.arrow, self.title_label, self.badge]:
            widget.bind("<Button-1>", self.toggle)
            widget.bind("<Enter>", self._on_header_enter)
            widget.bind("<Leave>", self._on_header_leave)

    def _on_header_enter(self, event=None):
        """Highlight header on hover"""
        hover_color = COLORS["bg_hover"]
        self.header.configure(bg=hover_color)
        self.arrow.configure(bg=hover_color)
        self.title_label.configure(bg=hover_color)
        self.badge.configure(bg=hover_color)

    def _on_header_leave(self, event=None):
        """Reset header color on mouse leave"""
        header_color = COLORS["bg_header"]
        self.header.configure(bg=header_color)
        self.arrow.configure(bg=header_color)
        self.title_label.configure(bg=header_color)
        self.badge.configure(bg=header_color)

    def toggle(self, event=None):
        """Toggle between expanded and collapsed states"""
        if self.expanded:
            self.content.pack_forget()
            self.arrow.config(text="▶")
        else:
            self.content.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 12))
            self.arrow.config(text="▼")
        self.expanded = not self.expanded

    def expand(self):
        """Expand the section if collapsed"""
        if not self.expanded:
            self.toggle()

    def collapse(self):
        """Collapse the section if expanded"""
        if self.expanded:
            self.toggle()

    def update_badge(self, filled=None, total=None):
        """Update the badge showing filled/total fields"""
        if self.badge_callback:
            filled, total = self.badge_callback()
        if filled is not None and total is not None:
            self.badge.config(text=f"[{filled}/{total}]")
        else:
            self.badge.config(text="")

    def get_content_frame(self):
        """Return the content frame where child widgets should be added"""
        return self.content
