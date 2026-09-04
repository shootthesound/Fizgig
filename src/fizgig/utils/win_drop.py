"""Files dragged from Explorer onto a Tk window, with WHERE they landed.

The same route Gizmo takes (gizmo.py::enable_file_drop) — the Windows shell API, not
tkinterdnd2, because a new dependency would mean everyone re-running the installer for a
convenience. This copy adds the drop point, so a window with several targets (the Repair
Studio's first / last frame slots) can tell which one the file was dropped on.

Tk has no drag-and-drop, so the window procedure is subclassed to catch WM_DROPFILES. THE RULE
THAT MAKES THIS SAFE: the procedure must not touch Tk. Not one call. Tk's mainloop sits inside
Tcl with the GIL released; a drop arrives, Windows dispatches into the window procedure, and a
`root.after` from there re-enters Tcl underneath the loop already running it — which killed the
interpreter outright in Gizmo's first version ("PyEval_RestoreThread: the function must be
called with the GIL held"). So the procedure only appends to a plain list, and an ordinary Tk
timer, running where Tk expects to be running, picks it up.

Not available off Windows (a RunPod pod is Linux): `enable_file_drop` returns False and the
caller keeps its Browse button.
"""

import os


def enable_file_drop(root, on_drop):
    """Accept files dragged onto `root` (a Tk or Toplevel). Returns True if it took.

    `on_drop(path, x_root, y_root)` is called on the Tk thread with the first dropped file and
    the drop point in SCREEN coordinates — `root.winfo_containing(x_root, y_root)` gives the
    widget under it. Explorer shows the drop cursor over the whole window; the caller decides
    what a drop anywhere else means (usually nothing).
    """
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        root.update_idletasks()
        # wm_frame is the REAL top-level window. winfo_id is Tk's child frame, which never
        # receives WM_DROPFILES however willing it looks.
        hwnd = int(root.wm_frame(), 16)
        user32, shell32 = ctypes.windll.user32, ctypes.windll.shell32
        WM_DROPFILES, GWLP_WNDPROC = 0x0233, -4

        # Pointer-sized by hand: WPARAM and the handle SetWindowLongPtrW returns do not fit
        # ctypes' default c_int, and the resulting OverflowError is raised inside the window
        # procedure where nothing can catch it.
        _64 = ctypes.sizeof(ctypes.c_void_p) == 8
        LONG_PTR = ctypes.c_longlong if _64 else ctypes.c_long
        UINT_PTR = ctypes.c_ulonglong if _64 else ctypes.c_ulong
        WNDPROC = ctypes.WINFUNCTYPE(LONG_PTR, ctypes.c_void_p, ctypes.c_uint,
                                     UINT_PTR, LONG_PTR)
        for fn, args, res in (
                (user32.CallWindowProcW,
                 [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, UINT_PTR, LONG_PTR], LONG_PTR),
                (user32.SetWindowLongPtrW,
                 [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p], ctypes.c_void_p),
                (user32.DefWindowProcW,
                 [ctypes.c_void_p, ctypes.c_uint, UINT_PTR, LONG_PTR], LONG_PTR),
                (user32.ClientToScreen, [ctypes.c_void_p, ctypes.c_void_p], wintypes.BOOL),
                (shell32.DragQueryFileW,
                 [ctypes.c_void_p, ctypes.c_uint, ctypes.c_wchar_p, ctypes.c_uint], ctypes.c_uint),
                (shell32.DragQueryPoint, [ctypes.c_void_p, ctypes.c_void_p], wintypes.BOOL),
                (shell32.DragFinish, [ctypes.c_void_p], None)):
            fn.argtypes, fn.restype = args, res

        shell32.DragAcceptFiles(ctypes.c_void_p(hwnd), True)
        inbox, old = [], [None]

        def proc(h, msg, wparam, lparam):
            # Plain Python only. No Tk, no Tcl, nothing that can raise if it can be helped — this
            # runs on every message the window receives.
            try:
                if msg == WM_DROPFILES:
                    try:
                        buf = ctypes.create_unicode_buffer(1024)
                        if shell32.DragQueryFileW(ctypes.c_void_p(wparam), 0, buf, 1024):
                            pt = wintypes.POINT()
                            shell32.DragQueryPoint(ctypes.c_void_p(wparam), ctypes.byref(pt))
                            user32.ClientToScreen(ctypes.c_void_p(h), ctypes.byref(pt))
                            inbox.append((buf.value, int(pt.x), int(pt.y)))
                    finally:
                        shell32.DragFinish(ctypes.c_void_p(wparam))
                    return 0
                return user32.CallWindowProcW(old[0], h, msg, wparam, lparam)
            except Exception:
                try:
                    return user32.DefWindowProcW(h, msg, wparam, lparam)
                except Exception:
                    return 0

        cb = WNDPROC(proc)
        old[0] = user32.SetWindowLongPtrW(ctypes.c_void_p(hwnd), GWLP_WNDPROC,
                                          ctypes.cast(cb, ctypes.c_void_p))
        if not old[0]:
            return False
        root._drop_callback = cb          # held, or it is collected and the window proc dangles

        def drain():
            while inbox:
                path, x, y = inbox.pop(0)
                try:
                    on_drop(path, x, y)
                except Exception:
                    pass
            root.after(120, drain)

        root.after(120, drain)
        return True
    except Exception:
        return False
