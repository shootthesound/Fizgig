import os as _os

# Refined Color Palette (Fizgig Visual Style Guide)
COLORS = {
    "bg_deep": "#1E2530",        # Main window background
    "bg_surface": "#252D38",     # Cards, panels, inputs
    "bg_hover": "#2A3542",       # Hover states
    "bg_header": "#1A2028",      # Collapsible section headers

    "text_primary": "#F0F4F8",   # Main text
    "text_secondary": "#8A9BAE", # Labels
    # Explanatory prose — card descriptions, row hints, fine print. Its own tier because the
    # muted grey below reads at 2.54:1 on a card, which fails WCAG even for large text, and this
    # is the copy that actually explains the app. 8.64:1, still a step down from text_primary so
    # a card title stays visibly louder than its explanation.
    "text_explain": "#C3CDD9",
    # Genuinely de-emphasised UI only: disabled controls, placeholders, and the one-word captions
    # beside widgets ("seed", "W", "H"). NOT for prose — that is text_explain above.
    "text_muted": "#5A6B7E",

    "accent": "#3B82F6",         # Primary actions, links
    "accent_hover": "#60A5FA",   # Accent hover
    "accent_subtle": "#1E3A5F",  # Accent backgrounds

    # Training-queue button (status bar, lower right). Deliberately a LIGHT blue: on a dark
    # bar a pale block reads as a distinct control, where another dark-surface panel just
    # blends into the furniture. Text on it is bg_deep (8.6:1) — text_primary would vanish.
    "queue_blue": "#93C5FD",
    "queue_blue_hover": "#BFDBFE",

    "border": "#3A4555",         # Borders, dividers
    "border_focus": "#3B82F6",   # Focus rings

    # Scrollbar thumb — the accent blue, matching the selected tab. Named entries rather than
    # pointing straight at "accent" so this stays tunable without dragging tabs and links along
    # with it. 4.19:1 against the trough, up from the 1.06:1 it used to be.
    "scrollbar_thumb": "#3B82F6",
    "scrollbar_thumb_hover": "#60A5FA",

    "success": "#10B981",        # Success states
    "warning": "#F59E0B",        # Warnings
    "error": "#EF4444",          # Errors
}

# Typography
FONT_FAMILY = "Segoe UI"
FONT_MONO = "Consolas"

# Legacy color constants (for backwards compatibility during transition)
BG_COLOR = COLORS["bg_deep"]
FG_COLOR = COLORS["text_primary"]
ACCENT_COLOR = COLORS["accent"]
ENTRY_BG = COLORS["bg_surface"]
BUTTON_ACTIVE = COLORS["bg_hover"]
BORDER_COLOR = COLORS["border"]
ACTIVE_ENTRY_BG = "white"  # Background color for active entry field
ACTIVE_ENTRY_FG = "black"  # Text color for active entry field

# Preview resolutions offered by BOTH the Samples tab and the live "Override next sample" panel.
# They used to be two hardcoded lists that had drifted — the Samples tab reached 1536 while the
# override stopped at 1024, so a run previewing at 1280+ could not be reproduced by the override,
# which silently downgraded it. Nothing downstream caps the value (Krea 2 rounds up to alignment,
# Klein floors to a multiple of 16), so the ceiling was purely this list.
SAMPLE_RESOLUTIONS = ["512", "640", "768", "1024", "1280", "1536"]

from fizgig_gui.core.paths import REPO_ROOT as _REPO_ROOT

# Directory for custom presets (per architecture)
PRESETS_DIR = _os.path.join(_REPO_ROOT, "presets")

# Snapshot of settings from the most recent training launch — restorable via "Load Last Train" button
LAST_TRAIN_FILE = _os.path.join(PRESETS_DIR, ".last_train_settings.json")
# Training queue: full settings snapshots waiting to run back-to-back. Survives restart;
# never auto-starts on launch (a queue found at startup waits for the user's first Start).
QUEUE_FILE = _os.path.join(PRESETS_DIR, "training_queue.json")

# Face detection imports (optional - graceful fallback if not installed)
try:
    from face_utils import (FaceDetector, FaceEmbedder, crop_to_face, draw_face_boxes,
                            is_face_detection_available)
    FACE_DETECTION_AVAILABLE = is_face_detection_available()
except ImportError:
    FACE_DETECTION_AVAILABLE = False
    FaceDetector = None
    FaceEmbedder = None
