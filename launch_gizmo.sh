#!/bin/bash
# Gizmo - cut training clips Fizgig will accept (the Linux twin of Launch Gizmo.bat)
cd "$(dirname "$0")"
if [ ! -x venv/bin/python ]; then
    echo "Fizgig's Python environment is missing - run the installer first."
    exit 1
fi
source venv/bin/activate
if ! python -c "import tkinter; tkinter.Tcl()" 2>/dev/null; then
    v=$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
    echo "Gizmo needs Tk, which Fizgig's Python doesn't have. Install it, then run this again:"
    echo "  Ubuntu / Debian: sudo apt install python3-tk   (or python${v}-tk)"
    echo "  Fedora: sudo dnf install python3-tkinter    Arch: sudo pacman -S tk"
    exit 1
fi
python src/gizmo/gizmo.py "$@" &
disown
