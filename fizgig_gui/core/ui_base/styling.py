import tkinter as tk
from tkinter import ttk

from fizgig_gui.core.config.constants import COLORS, FONT_FAMILY, BG_COLOR, ACTIVE_ENTRY_BG, ACTIVE_ENTRY_FG


class StylingMixin:
    def setup_styles(self):
        """Set up styles for refined dark theme (Fizgig Visual Style Guide)"""
        style = ttk.Style()
        style.theme_use("clam")

        # Base styles with new palette
        style.configure(".",
            background=COLORS["bg_deep"],
            foreground=COLORS["text_primary"],
            font=(FONT_FAMILY, 10)
        )
        style.configure("TFrame", background=COLORS["bg_deep"])
        style.configure("TLabel",
            background=COLORS["bg_deep"],
            foreground=COLORS["text_primary"],
            font=(FONT_FAMILY, 10)
        )

        # Surface style for cards/panels
        style.configure("Surface.TFrame", background=COLORS["bg_surface"])

        # Green render-progress bar (Repair Studio) — matches the Start button's green.
        style.configure("Green.Horizontal.TProgressbar",
                        background="#2E8B57", troughcolor=COLORS["bg_deep"],
                        bordercolor=COLORS["bg_deep"],
                        lightcolor="#2E8B57", darkcolor="#2E8B57")

        # Collapsible header style
        style.configure("CollapsibleHeader.TFrame", background=COLORS["bg_header"])

        # Section header label style
        style.configure("SectionHeader.TLabel",
            background=COLORS["bg_deep"],
            foreground=COLORS["text_primary"],
            font=(FONT_FAMILY, 12, "bold")
        )

        # Secondary label style (for field labels)
        style.configure("Secondary.TLabel",
            background=COLORS["bg_deep"],
            foreground=COLORS["text_secondary"],
            font=(FONT_FAMILY, 10)
        )

        # Default button (Secondary style)
        style.configure(
            "TButton",
            background=COLORS["bg_surface"],
            foreground=COLORS["text_primary"],
            bordercolor=COLORS["border"],
            borderwidth=1,
            focusthickness=2,
            focuscolor=COLORS["accent"],
            padding=[16, 8],
            font=(FONT_FAMILY, 10, "bold")
        )
        style.map(
            "TButton",
            background=[("active", COLORS["bg_hover"]), ("pressed", COLORS["bg_hover"])],
            foreground=[("active", COLORS["text_primary"]), ("pressed", COLORS["text_primary"])]
        )

        # Primary button (accent color)
        style.configure(
            "Primary.TButton",
            background=COLORS["accent"],
            foreground="white",
            bordercolor=COLORS["accent"],
            borderwidth=1,
            focusthickness=2,
            focuscolor=COLORS["accent_hover"],
            padding=[16, 8],
            font=(FONT_FAMILY, 10, "bold")
        )
        style.map(
            "Primary.TButton",
            background=[("active", COLORS["accent_hover"]), ("pressed", COLORS["accent_hover"])],
            foreground=[("active", "white"), ("pressed", "white")]
        )

        # Danger button (for stop actions)
        style.configure(
            "Danger.TButton",
            background=COLORS["error"],
            foreground="white",
            bordercolor=COLORS["error"],
            borderwidth=1,
            focusthickness=2,
            focuscolor=COLORS["error"],
            padding=[16, 8],
            font=(FONT_FAMILY, 10, "bold")
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#DC2626"), ("pressed", "#DC2626")],
            foreground=[("active", "white"), ("pressed", "white")]
        )

        # Checkbutton
        style.configure("TCheckbutton",
            background=COLORS["bg_deep"],
            foreground=COLORS["text_primary"],
            font=(FONT_FAMILY, 10)
        )
        style.map("TCheckbutton",
            background=[("active", COLORS["bg_deep"])],
            foreground=[("active", COLORS["text_primary"])]
        )
        style.configure("TRadiobutton",
            background=COLORS["bg_deep"],
            foreground=COLORS["text_primary"],
            font=(FONT_FAMILY, 10)
        )
        style.map("TRadiobutton",
            background=[("active", COLORS["bg_deep"])],
            foreground=[("active", COLORS["text_primary"])]
        )
        style.configure("Surface.TRadiobutton",
            background=COLORS["bg_surface"],
            foreground=COLORS["text_primary"],
            font=(FONT_FAMILY, 10)
        )
        style.map("Surface.TRadiobutton",
            background=[("active", COLORS["bg_surface"])],
            foreground=[("active", COLORS["text_primary"])]
        )

        # Surface checkbutton (for use inside collapsible sections)
        style.configure("Surface.TCheckbutton",
            background=COLORS["bg_surface"],
            foreground=COLORS["text_primary"],
            font=(FONT_FAMILY, 10)
        )
        style.map("Surface.TCheckbutton",
            background=[("active", COLORS["bg_surface"])],
            foreground=[("active", COLORS["text_primary"])]
        )

        # Notebook (tabs)
        style.configure("TNotebook",
            background=COLORS["bg_deep"],
            borderwidth=0
        )
        style.configure("TNotebook.Tab",
            background=COLORS["bg_surface"],
            foreground=COLORS["text_primary"],
            padding=[12, 6],
            font=(FONT_FAMILY, 11, "bold")
        )
        style.map("TNotebook.Tab",
            background=[("selected", COLORS["accent"])],
            foreground=[("selected", "white")]
        )

        # Entry field — explicit insert cursor settings so the caret is visible on click
        style.configure(
            "TEntry",
            fieldbackground=COLORS["bg_surface"],
            foreground=COLORS["text_primary"],
            bordercolor=COLORS["border"],
            insertcolor=COLORS["accent"],
            insertwidth=2,
            font=(FONT_FAMILY, 10)
        )
        style.map("TEntry",
            fieldbackground=[("focus", ACTIVE_ENTRY_BG)],
            foreground=[("focus", ACTIVE_ENTRY_FG)],
            bordercolor=[("focus", COLORS["border_focus"])],
            insertcolor=[("focus", COLORS["accent"])],
        )

        # Combobox — explicit insert cursor settings so editable comboboxes show a caret
        style.configure(
            "TCombobox",
            fieldbackground=COLORS["bg_surface"],
            background=COLORS["bg_surface"],
            foreground=COLORS["text_primary"],
            bordercolor=COLORS["border"],
            arrowcolor=COLORS["text_secondary"],
            insertcolor=COLORS["accent"],
            insertwidth=2,
            font=(FONT_FAMILY, 10)
        )
        style.map("TCombobox",
            fieldbackground=[("focus", ACTIVE_ENTRY_BG), ("readonly", COLORS["bg_surface"]), ("!disabled", COLORS["bg_surface"])],
            foreground=[("focus", ACTIVE_ENTRY_FG), ("readonly", COLORS["text_primary"]), ("!disabled", COLORS["text_primary"])],
            selectbackground=[("readonly", COLORS["bg_surface"]), ("!disabled", COLORS["bg_surface"])],
            selectforeground=[("readonly", COLORS["text_primary"]), ("!disabled", COLORS["text_primary"])],
            bordercolor=[("focus", COLORS["border_focus"])]
        )

        # Treeview (the Metadata tab's custom-fields table). Nothing styled this before it —
        # "clam" falls back to a stock white row background with no matching foreground, so
        # rows rendered as near-invisible light-grey-on-white against an otherwise dark app.
        style.configure(
            "Treeview",
            background=COLORS["bg_surface"],
            fieldbackground=COLORS["bg_surface"],
            foreground=COLORS["text_primary"],
            bordercolor=COLORS["border"],
            font=(FONT_FAMILY, 10),
            rowheight=24,
        )
        style.map("Treeview",
            background=[("selected", COLORS["accent"])],
            foreground=[("selected", "white")]
        )
        style.configure(
            "Treeview.Heading",
            background=COLORS["bg_header"],
            foreground=COLORS["text_primary"],
            font=(FONT_FAMILY, 10, "bold"),
        )
        style.map("Treeview.Heading",
            background=[("active", COLORS["bg_hover"])]
        )

        # Scrollbar. `background` is the thumb — the part you drag — and it used to be bg_header
        # on a bg_deep trough, which is 1.06:1. Not "subtle": indistinguishable from the track,
        # so on a tall tab there was no visible clue that the panel scrolled at all.
        #
        # The accent blue, same as the selected tab. Colour lives in COLORS as its own entry so
        # the scrollbar can be retuned without touching what "accent" means everywhere else.
        for _orient in ("Vertical", "Horizontal"):
            style.configure(
                f"{_orient}.TScrollbar",
                background=COLORS["scrollbar_thumb"],
                troughcolor=COLORS["bg_deep"],
                bordercolor=COLORS["border"],
                arrowcolor=COLORS["text_primary"],
                darkcolor=COLORS["bg_deep"],
                lightcolor=COLORS["bg_deep"],
                width=12
            )
            style.map(
                f"{_orient}.TScrollbar",
                background=[("active", COLORS["scrollbar_thumb_hover"]),
                            ("pressed", COLORS["scrollbar_thumb_hover"])]
            )

        # LabelFrame
        style.configure("TLabelframe",
            background=COLORS["bg_deep"],
            bordercolor=COLORS["border"]
        )
        style.configure("TLabelframe.Label",
            background=COLORS["bg_deep"],
            foreground=COLORS["text_primary"],
            font=(FONT_FAMILY, 10, "bold")
        )

    def create_scrollable_frame(self, parent):
        """Create a scrollable frame within a parent widget"""
        # Create a canvas
        canvas = tk.Canvas(parent, bg=BG_COLOR, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        # Configure the scrollable frame
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        # Create window inside canvas
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        # Configure canvas to expand horizontally
        def configure_canvas(event):
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind("<Configure>", configure_canvas)
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack the canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mousewheel: handled by the global router (_route_mousewheel) — it finds this
        # canvas through the pointer's ancestry, so no per-tab bind_all is needed (the old
        # Enter/Leave bind_all dance is what made scrollable areas fight each other).

        return scrollable_frame, canvas