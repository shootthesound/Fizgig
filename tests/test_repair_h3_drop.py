"""A photo dragged from Explorer onto the Repair Studio's first / last frame slot (MiniMax H3),
plus the remembered Browse folders.

A REAL WM_DROPFILES with a drop point, POSTED from another thread and left to land on an idle
mainloop — the condition a real drag creates and a SendMessage from inside root.update() never
does (see tests/test_gizmo_drop.py for the crash that taught us that). The window has to be on
screen: the drop point is resolved with winfo_containing, which only sees viewable widgets.

Run: FIZGIG_NO_PERSIST=1 venv/Scripts/python.exe tests/test_repair_h3_drop.py
"""

import ctypes
import os
import sys
import tempfile
import threading
import tkinter as tk
from ctypes import wintypes

os.environ["FIZGIG_NO_PERSIST"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image                                                        # noqa: E402

import lora_trainer_gui as G                                                 # noqa: E402

G.LoRATrainerGUI.save_prefs = lambda self, *a, **k: None
G.LoRATrainerGUI._save_training_queue = lambda self, *a, **k: None
G.LoRATrainerGUI._save_last_used_paths = lambda self, *a, **k: None
threading.excepthook = lambda _a: None
_fails = []


def ck(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (f"  {detail}" if detail else ""), flush=True)
    if not ok:
        _fails.append(name)


TMP = tempfile.mkdtemp(prefix="fz_kfdrop_")
png = os.path.join(TMP, "photo.png")
Image.new("RGB", (96, 96), (200, 120, 80)).save(png)
txt = os.path.join(TMP, "notes.txt")
open(txt, "w").write("not a photo")

root = tk.Tk()
root.geometry("1500x1000+0+0")
root.attributes("-topmost", True)          # winfo_containing must see OUR widgets at the point
app = G.LoRATrainerGUI(root)
app.notebook.select(app.repair_studio_tab)
app.repair_family_var.set("minimax")
app._on_repair_family_changed()
root.update()

ck("drops are accepted (Windows)", app._repair_kf_can_drop or os.name != "nt",
   f"can_drop={app._repair_kf_can_drop}")
ck("the window-proc callback is held", hasattr(root, "_drop_callback") or not app._repair_kf_can_drop)

# --- remembered Browse folders (no window messages needed) ------------------------------------
app.last_used["lora_browse_dir"] = TMP
ck("LoRA Browse opens where the last LoRA was picked",
   os.path.normcase(app._lora_initialdir()) == os.path.normcase(TMP), app._lora_initialdir())
app.last_used["lora_browse_dir"] = os.path.join(TMP, "gone")
ck("...a vanished folder falls through to the pref / output-dir rule",
   os.path.normcase(app._lora_initialdir()) != os.path.normcase(os.path.join(TMP, "gone")))
app._remember_browse_dir("lora_browse_dir", png)
ck("picking a file remembers its folder",
   os.path.normcase(app.last_used.get("lora_browse_dir", "")) == os.path.normcase(TMP))

if app._repair_kf_can_drop:
    # the render is not the subject here, and the crop dialog is modal — stand both down
    app._on_preview_param_changed = lambda *a, **k: None
    crops = []
    app._repair_kf_crop_dialog = lambda path, initial=None: (crops.append(path), (0, 0, 48, 48))[1]

    class DROPFILES(ctypes.Structure):
        _fields_ = [("pFiles", wintypes.DWORD), ("pt", wintypes.POINT),
                    ("fNC", wintypes.BOOL), ("fWide", wintypes.BOOL)]

    k32, u32 = ctypes.windll.kernel32, ctypes.windll.user32
    k32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    k32.GlobalAlloc.restype = ctypes.c_void_p
    k32.GlobalLock.argtypes = [ctypes.c_void_p]
    k32.GlobalLock.restype = ctypes.c_void_p
    k32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    u32.PostMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_ulonglong, ctypes.c_longlong]
    u32.ScreenToClient.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    hwnd = int(root.wm_frame(), 16)

    def make_hdrop(path, sx, sy):
        """A real HDROP: header (with the drop point in the window's CLIENT coordinates, as the
        shell fills it in), the path, and the double null it terminates with."""
        pt = wintypes.POINT(int(sx), int(sy))
        u32.ScreenToClient(ctypes.c_void_p(hwnd), ctypes.byref(pt))
        blob = bytes(ctypes.sizeof(DROPFILES)) + (path + "\0\0").encode("utf-16-le")
        handle = k32.GlobalAlloc(0x0002, len(blob))
        ptr = k32.GlobalLock(handle)
        ctypes.memmove(ptr, blob, len(blob))
        header = DROPFILES.from_address(ptr)
        header.pFiles = ctypes.sizeof(DROPFILES)
        header.pt = pt
        header.fWide = True
        k32.GlobalUnlock(handle)
        return handle

    def drop(path, sx, sy, wait=1.2):
        hdrop = make_hdrop(path, sx, sy)
        threading.Timer(0.3, lambda: u32.PostMessageW(ctypes.c_void_p(hwnd), 0x0233, hdrop, 0)).start()
        threading.Timer(wait, root.quit).start()
        root.mainloop()

    def thumb_point(slot):
        w = app._repair_h3_kf_widgets[slot]["thumb"]
        root.update()
        return w.winfo_rootx() + w.winfo_width() // 2, w.winfo_rooty() + w.winfo_height() // 2

    thumb = app._repair_h3_kf_widgets["first"]["thumb"]
    ck("the empty slot invites a drop", "drop" in thumb.cget("text"), thumb.cget("text"))
    x, y = thumb_point("first")
    ck("the first-frame slot is on screen", thumb.winfo_viewable() and app._repair_kf_slot_at(x, y) == "first",
       f"viewable={thumb.winfo_viewable()} slot_at={app._repair_kf_slot_at(x, y)}")

    drop(png, x, y)
    ck("a photo dropped on the first slot is cropped and stored",
       app._repair_h3_kf.get("first") == {"path": png, "rect": (0, 0, 48, 48)} and crops == [png],
       str(app._repair_h3_kf.get("first")))
    ck("...and its folder is remembered for the photo Browse",
       os.path.normcase(app.last_used.get("kf_browse_dir", "")) == os.path.normcase(TMP))
    ck("...and the thumb shows it", app._repair_h3_kf_widgets["first"]["thumb"].image is not None)

    x2, y2 = thumb_point("last")
    drop(png, x2, y2)
    ck("a photo dropped on the last slot lands in the last slot",
       app._repair_h3_kf.get("last") == {"path": png, "rect": (0, 0, 48, 48)} and len(crops) == 2)

    app._repair_h3_kf["first"] = None
    app._repair_kf_refresh_thumbs()
    n = len(crops)
    app.repair_status_var.set("")
    drop(txt, x, y)
    ck("a non-image on a slot is refused with a status line, no crop dialog",
       app._repair_h3_kf.get("first") is None and len(crops) == n
       and "takes a photo" in app.repair_status_var.get(), app.repair_status_var.get())

    # somewhere that is not a slot: the tab banner, top-left of the window
    bx, by = root.winfo_rootx() + 30, root.winfo_rooty() + 30
    ck("the banner point is not a slot", app._repair_kf_slot_at(bx, by) is None)
    drop(png, bx, by)
    ck("a drop anywhere else does nothing", app._repair_h3_kf.get("first") is None and len(crops) == n)

    ck("the app is still alive after four posted drops", bool(root.winfo_exists()))

root.destroy()
print()
if _fails:
    print(f"{len(_fails)} FAILED: " + ", ".join(_fails))
    sys.exit(1)
print("ALL PASS")
