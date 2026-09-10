"""Clips confined to the likeness blocks whenever Optimised Likeness Learning is on
(29 Aug field-proven; the 'Restrict video' sub-tick was retired 7 Sep — no extra tick).

Pins the silent-failure surfaces: no restrict-video widget exists any more; the builder
must emit --clip_blocks exactly when likeness is on, LoRA and FT alike, and never when it
is off; the settings key MINIMAX_FT_CLIP_LIKENESS is gone from the launch dict.

Run: venv/Scripts/python.exe tests/test_minimax_ft_clip_restrict.py
"""
import hashlib
import os
import sys

os.environ["FIZGIG_NO_PERSIST"] = "1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import tkinter as tk  # noqa: E402
import lora_trainer_gui as G  # noqa: E402

G.LAST_USED_FILE = os.path.join(os.environ.get("TEMP", "/tmp"), "nope", ".last_used.json")
G.LoRATrainerGUI.save_prefs = lambda self, *a, **k: None
G.LoRATrainerGUI._save_training_queue = lambda self, *a, **k: None

_WATCH = [os.path.join(REPO, "prefs.json"), os.path.join(REPO, "last_used.json")]


def _digests():
    out = {}
    for p in _WATCH:
        try:
            with open(p, "rb") as fh:
                out[p] = hashlib.sha256(fh.read()).hexdigest()
        except FileNotFoundError:
            out[p] = None
    return out


before = _digests()
FAILS = []


def ck(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


root = tk.Tk()
root.geometry("1400x900+3000+3000")   # offscreen but MAPPED — withdrawn roots unmap children
app = G.LoRATrainerGUI(root)
root.update()

H3 = "MiniMax H3 (experimental)"
assert H3 in G.ARCHITECTURES


def mapped(w):
    try:
        return bool(w.winfo_ismapped())
    except Exception:
        return False


def settle(likeness=None, ft=None, arch=H3):
    app.notebook.select(app.training_tab)
    app.architecture_var.set(arch)
    app.update_ui_for_architecture()
    if likeness is not None:
        app.entries["MINIMAX_LIKENESS_MODE"].set(
            G.MINIMAX_MODE_FAST if likeness else G.MINIMAX_MODE_OFF)
    if ft is not None and hasattr(app, "minimax_finetune_var"):
        app.minimax_finetune_var.set(ft)
        app._on_minimax_ft_toggle()
    app._sync_minimax_likeness_state()
    root.update()


# --- no sub-tick any more (running app, not source) ----------------------------------------
settle(likeness=True, ft=True)
ck("no restrict-video widget exists", not hasattr(app, "_minimax_ft_clip_cb")
   and "MINIMAX_FT_CLIP_LIKENESS" not in app.entries)
ck("likeness hint says photos AND clips train the identity blocks",
   "photos and clips train the identity blocks" in app._minimax_likeness_hint.cget("text").lower(),
   app._minimax_likeness_hint.cget("text"))
settle(likeness=True, ft=False)

# --- builder: --clip_blocks exactly when likeness is on -----------------------------------
BASE = {
    "DATASET_CONFIG": "X:/ds.toml", "LORA_OUTPUT_DIR": "X:/out", "LORA_NAME": "t",
    "NETWORK_DIM": 16, "NETWORK_ALPHA": 16, "LEARNING_RATE": 1e-4, "MAX_TRAIN_EPOCHS": 20,
    "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42, "BLOCKS_SWAP": "auto", "ADAPTIVE_LR": False,
    "MAX_GRAD_NORM": "1.0", "OPTIMIZER_TYPE": "adamw8bit", "OPTIMIZER_ARGS": "",
    "METADATA_TITLE": "", "METADATA_AUTHOR": "", "METADATA_DESCRIPTION": "",
    "METADATA_LICENSE": "", "METADATA_TAGS": "", "METADATA_TRIGGER_PHRASE": "",
    "MINIMAX_LIKENESS_MODE": G.MINIMAX_MODE_FAST, "MINIMAX_BLOCKS": "all",
}
CFG = G.ARCHITECTURES[H3]


def cmd_of(**over):
    app.settings = dict(BASE, **over)
    return [str(x) for x in app.build_training_command(CFG)]


def has_clip_blocks(c):
    return "--clip_blocks" in c and c[c.index("--clip_blocks") + 1] == G.MINIMAX_LIKENESS_BLOCKS


app.minimax_finetune_var.set(True)
c = cmd_of()
ck("FT + likeness -> --clip_blocks 20-49 emitted", has_clip_blocks(c))
ck("...and --photo_blocks still travels", "--photo_blocks" in c)
c = cmd_of(MINIMAX_FT_CLIP_LIKENESS=False)
ck("a stale saved MINIMAX_FT_CLIP_LIKENESS=False changes nothing", has_clip_blocks(c))
c = cmd_of(MINIMAX_LIKENESS_MODE=G.MINIMAX_MODE_OFF)
ck("FT, likeness OFF -> no --clip_blocks (rides likeness only)", "--clip_blocks" not in c)
app.minimax_finetune_var.set(False)
c = cmd_of()
ck("LoRA mode + likeness -> --clip_blocks too", has_clip_blocks(c))
c = cmd_of(MINIMAX_LIKENESS_MODE=G.MINIMAX_MODE_OFF)
ck("LoRA mode, likeness OFF -> none", "--clip_blocks" not in c)
ck("launch dict no longer carries MINIMAX_FT_CLIP_LIKENESS",
   "MINIMAX_FT_CLIP_LIKENESS" not in open(os.path.join(REPO, "lora_trainer_gui.py"), encoding="utf-8").read())

# --- Checkpoint to LoRA button (29 Aug: pod users only have the browser GUI) -------------
settle(likeness=True, ft=True)
ck("H3 + FT: Checkpoint to LoRA button MAPPED", mapped(app._minimax_c2l_btn))
settle(likeness=True, ft=False)
ck("FT off: button hidden with the FT frame", not mapped(app._minimax_c2l_btn))
settle(likeness=True, ft=True)

import subprocess as _sp
_calls = []
_real_popen = _sp.Popen
_sp.Popen = lambda argv, **kw: _calls.append((argv, kw)) or None
try:
    app._launch_diff_to_lora()
finally:
    _sp.Popen = _real_popen
ck("launcher spawns diff_to_lora_gui.py", _calls
   and str(_calls[0][0][-1]).endswith("diff_to_lora_gui.py"), _calls)
ck("launcher cwd is the Fizgig folder", _calls and _calls[0][1].get("cwd") == G.FIZGIG_DIR)
# --- nothing of the user's was touched ---------------------------------------------------
after = _digests()
ck("prefs.json / last_used.json byte-identical after the run", before == after,
   [p for p in before if before[p] != after[p]])

try:
    root.destroy()
except Exception:
    pass

print()
if FAILS:
    print(f"FAILED: {len(FAILS)} pin(s):")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("ALL PASS")
