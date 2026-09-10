r"""Training mode (Fast / Ultra quality / Off) — the GUI wiring. Was a checkbox called
Optimised Likeness Learning until 10 Sep 2026.

The silent failure modes this pins: the mode not reaching the launch dict (the hand-curated
settings.update trap), the Style preset inheriting Fast and having its 0-3,6-47 blocks spec
silently ignored, a preset writing the blocks value into a DISABLED combobox and losing it, a
stale typo in the greyed box blocking a launch it has no say in, and Ultra quietly emitting the
Fast flags (or nothing at all).

Run: venv\Scripts\python.exe tests\test_minimax_likeness_gui.py
"""
import hashlib
import os
import sys

os.environ["FIZGIG_NO_PERSIST"] = "1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

PREFS = os.path.join(REPO, "prefs.json")
_hash0 = hashlib.sha256(open(PREFS, "rb").read()).hexdigest() if os.path.exists(PREFS) else None

import tkinter as tk  # noqa: E402

import lora_trainer_gui as G  # noqa: E402

FAILS = []


def ck(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


G.LAST_USED_FILE = os.path.join(os.environ.get("TEMP", "/tmp"), "nope", ".last_used.json")
G.LoRATrainerGUI.save_prefs = lambda self, *a, **k: None
G.LoRATrainerGUI._save_training_queue = lambda self, *a, **k: None

root = tk.Tk()
root.withdraw()
app = G.LoRATrainerGUI(root)
app.architecture_var.set("MiniMax H3 (experimental)")
app.update_ui_for_architecture()
root.update_idletasks()

STYLE_KEY = "✨ MiniMax H3 Style (LoRA 8)"
# Looked up by PREFIX, not spelled out: this pin died on a KeyError when the preset was
# renamed 40 -> 50 epochs, which is a rename the test has no opinion about. What it cares
# about is "the Fast preset", so ask for that.
FAST_KEY = next(k for k in G.MINIMAX_BUILT_IN_PRESETS if k.startswith("✨ MiniMax H3 Fast"))
DEFAULTS_KEY = next(iter(G.MINIMAX_BUILT_IN_PRESETS))

# --- 0. the var: exists, defaults ON, rides the preset machinery -----------------------------------
ck("mode var exists in self.entries", "MINIMAX_LIKENESS_MODE" in app.entries)
ck("defaults to Fast", G.minimax_likeness_mode(app.entries["MINIMAX_LIKENESS_MODE"].get()) == "fast",
   app.entries["MINIMAX_LIKENESS_MODE"].get())
ck("three modes offered", len(G.MINIMAX_LIKENESS_MODE_OPTIONS) == 3
   and [G.minimax_likeness_mode(o) for o in G.MINIMAX_LIKENESS_MODE_OPTIONS] == ["fast", "ultra", "off"])
ck("collected into presets/snapshots",
   "MINIMAX_LIKENESS_MODE" in app._collect_preset_values())
ck("preset dicts: Defaults Fast / Fast Fast / Style Off",
   G.minimax_likeness_mode(G.MINIMAX_BUILT_IN_PRESETS[DEFAULTS_KEY]["MINIMAX_LIKENESS_MODE"]) == "fast"
   and G.minimax_likeness_mode(G.MINIMAX_BUILT_IN_PRESETS[FAST_KEY]["MINIMAX_LIKENESS_MODE"]) == "fast"
   and G.minimax_likeness_mode(G.MINIMAX_BUILT_IN_PRESETS[STYLE_KEY]["MINIMAX_LIKENESS_MODE"]) == "off")

# --- 1. visibility: on the MiniMax tab, gone under Klein --------------------------------------------
ck("dropdown visible under MiniMax", bool(app._minimax_likeness_frame.winfo_manager())
   and bool(app._minimax_likeness_label.winfo_manager()))
app.architecture_var.set("Flux 2 Klein 9B")
app.update_ui_for_architecture()
root.update_idletasks()
ck("hidden under Klein", not app._minimax_likeness_frame.winfo_manager())
ck("Klein leaves Blocks to Train un-greyed even in Fast",
   str(app.entries["MINIMAX_BLOCKS"].cget("state")) != "disabled")
app.architecture_var.set("MiniMax H3 (experimental)")
app.update_ui_for_architecture()
root.update_idletasks()

# --- 2. greying: Fast and Ultra own the blocks row, Off hands it back ------------------------------
combo = app.entries["MINIMAX_BLOCKS"]
app.entries["MINIMAX_LIKENESS_MODE"].set(G.MINIMAX_MODE_FAST)
root.update_idletasks()
ck("Fast -> combobox disabled", str(combo.cget("state")) == "disabled")
ck("Fast -> hint says who disabled it and with what",
   "Training mode" in app._minimax_blocks_hint.cget("text")
   and G.MINIMAX_LIKENESS_BLOCKS in app._minimax_blocks_hint.cget("text"))
ck("Fast -> mode hint names the windows",
   G.MINIMAX_LIKENESS_BLOCKS in app._minimax_likeness_hint.cget("text")
   and G.MINIMAX_AUDIO_BLOCKS in app._minimax_likeness_hint.cget("text"))
combo_before = combo.get()
app.entries["MINIMAX_LIKENESS_MODE"].set(G.MINIMAX_MODE_ULTRA)
root.update_idletasks()
ck("Ultra -> combobox still disabled", str(combo.cget("state")) == "disabled")
ck("Ultra -> hint and readout say 6-49",
   G.MINIMAX_FULL_MODEL_BLOCKS in app._minimax_blocks_hint.cget("text")
   and G.MINIMAX_FULL_MODEL_BLOCKS in app._minimax_blocks_count.cget("text"))
app.entries["MINIMAX_LIKENESS_MODE"].set(G.MINIMAX_MODE_OFF)
root.update_idletasks()
ck("Off -> combobox editable again", str(combo.cget("state")) != "disabled")
ck("Off -> measured-answers hint restored",
   "Measured answers" in app._minimax_blocks_hint.cget("text"))
ck("combobox VALUE survives the Fast/Ultra/Off round-trip", combo.get() == combo_before)

# --- 3. the Style preset: False lands AND its blocks spec survives the disabled window -------------
app.entries["MINIMAX_LIKENESS_MODE"].set(G.MINIMAX_MODE_FAST)   # worst case: greyed on arrival
app._apply_preset_values(G.MINIMAX_BUILT_IN_PRESETS[STYLE_KEY])
root.update_idletasks()
ck("Style preset sets the mode to Off",
   G.minimax_likeness_mode(app.entries["MINIMAX_LIKENESS_MODE"].get()) == "off")
ck("Style preset's blocks spec landed despite the apply-order",
   G.minimax_block_spec(combo.get()) == "0-3, 6-47", repr(combo.get()))
ck("and the combobox is editable after it", str(combo.cget("state")) != "disabled")
app._apply_preset_values(G.MINIMAX_BUILT_IN_PRESETS[FAST_KEY])
root.update_idletasks()
ck("Fast preset sets Fast again and re-greys",
   G.minimax_likeness_mode(app.entries["MINIMAX_LIKENESS_MODE"].get()) == "fast"
   and str(combo.cget("state")) == "disabled")

# --- 4. validation: a stale typo in the greyed box must not block a launch -------------------------
app.entries["MINIMAX_LIKENESS_MODE"].set(G.MINIMAX_MODE_OFF)
combo.set("49-20")                                     # backwards range = a real parse error
_shown = []
_real = G.messagebox.showerror
G.messagebox.showerror = lambda title, msg, **k: _shown.append(msg)
try:
    app.validate_inputs()
    ck("Off: the typo IS caught", any("Blocks to Train" in m for m in _shown))
    _shown.clear()
    app.entries["MINIMAX_LIKENESS_MODE"].set(G.MINIMAX_MODE_FAST)
    app.validate_inputs()
    ck("Fast: the same typo is ignored (the box is owned by the mode)",
       not any("Blocks to Train" in m for m in _shown))
finally:
    G.messagebox.showerror = _real
combo.set("all")

# --- 5. the command: --photo_blocks when on, --train_blocks when off -------------------------------
BASE = {
    "DATASET_CONFIG": "X:/ds.toml", "LORA_OUTPUT_DIR": "X:/out", "LORA_NAME": "t",
    "NETWORK_DIM": 16, "NETWORK_ALPHA": 16, "LEARNING_RATE": 1e-4, "MAX_TRAIN_EPOCHS": 20,
    "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42, "BLOCKS_SWAP": "auto", "ADAPTIVE_LR": False,
    "MAX_GRAD_NORM": "1.0", "OPTIMIZER_TYPE": "adamw8bit", "OPTIMIZER_ARGS": "",
    "METADATA_TITLE": "", "METADATA_AUTHOR": "", "METADATA_DESCRIPTION": "",
    "METADATA_LICENSE": "", "METADATA_TAGS": "", "METADATA_TRIGGER_PHRASE": "",
}
CFG = G.ARCHITECTURES["MiniMax H3 (experimental)"]
app.prefs_vars["minimax_dit"].set("X:/fl2va.safetensors")
app.prefs_vars["minimax_vae"].set("X:/vae.safetensors")
app.prefs_vars["minimax_text_encoder"].set("X:/te.safetensors")


def cmd_of(**over):
    app.settings = dict(BASE, **over)
    return [str(x) for x in app.build_training_command(CFG)]


c = cmd_of(MINIMAX_LIKENESS_MODE=G.MINIMAX_MODE_FAST, MINIMAX_BLOCKS="all")
ck("Fast: --photo_blocks + --clip_blocks 20-49, voice 34-49", "--photo_blocks" in c
   and c[c.index("--photo_blocks") + 1] == G.MINIMAX_LIKENESS_BLOCKS
   and c[c.index("--clip_blocks") + 1] == G.MINIMAX_LIKENESS_BLOCKS
   and c[c.index("--audio_blocks") + 1] == G.MINIMAX_AUDIO_BLOCKS)
ck("Fast: no --train_blocks", "--train_blocks" not in c)
# Ultra is one range for every step type. The launch dict is what puts 6-49 into MINIMAX_BLOCKS,
# so the builder sees it there — pinned the same way a real launch produces it.
c = cmd_of(MINIMAX_LIKENESS_MODE=G.MINIMAX_MODE_ULTRA, MINIMAX_BLOCKS=G.MINIMAX_FULL_MODEL_BLOCKS)
ck("Ultra: --train_blocks 6-49", "--train_blocks" in c
   and c[c.index("--train_blocks") + 1] == G.MINIMAX_FULL_MODEL_BLOCKS)
ck("Ultra: no per-modality masks (one range for everything)",
   "--photo_blocks" not in c and "--clip_blocks" not in c and "--audio_blocks" not in c)
c = cmd_of(MINIMAX_LIKENESS_MODE=G.MINIMAX_MODE_OFF, MINIMAX_BLOCKS="27, 29-49")
ck("Off: --train_blocks flows as before", "--train_blocks" in c
   and c[c.index("--train_blocks") + 1] == "27, 29-49")
ck("Off: no --photo_blocks", "--photo_blocks" not in c)
c = cmd_of(MINIMAX_BLOCKS="all")
ck("no key at all (old settings file): falls back to Fast", "--photo_blocks" in c)

# --- 5b. a run saved BEFORE the dropdown restores as the mode it actually was -----------------
# Ticked was exactly Fast; unticked meant "Blocks to Train rules", which is Off — and the saved
# spec travels with it. Ultra is new, so no saved run was ever Ultra: mapping unticked to it
# would overwrite the user's own spec and change what they trained. Without the mapping the key
# is simply absent from self.entries and the whole choice is dropped in SILENCE.
for _old, _want, _spec in ((True, "fast", "all"), (False, "off", "0-3, 6-47")):
    app.entries["MINIMAX_LIKENESS_MODE"].set(G.MINIMAX_MODE_ULTRA)      # anything but the answer
    app._apply_preset_values({"MINIMAX_LIKENESS_OPT": _old, "MINIMAX_BLOCKS": _spec})
    root.update_idletasks()
    ck(f"old saved run, likeness={_old} -> mode {_want}",
       G.minimax_likeness_mode(app.entries["MINIMAX_LIKENESS_MODE"].get()) == _want,
       app.entries["MINIMAX_LIKENESS_MODE"].get())
    ck(f"...and its blocks spec survives ({_spec})",
       G.minimax_block_spec(combo.get()) == _spec, combo.get())
# and the same for a queued run, which launches straight from its own settings dict
_q = dict(BASE, MINIMAX_LIKENESS_OPT=True, MINIMAX_BLOCKS="all")
app.settings = _q
ck("old queued run, likeness=True -> the Fast flags",
   "--photo_blocks" in [str(x) for x in app.build_training_command(CFG)])
app.settings = dict(BASE, MINIMAX_LIKENESS_OPT=False, MINIMAX_BLOCKS="0-3, 6-47")
_c = [str(x) for x in app.build_training_command(CFG)]
ck("old queued run, likeness=False -> its own spec, no Fast flags",
   "--train_blocks" in _c and _c[_c.index("--train_blocks") + 1] == "0-3, 6-47"
   and "--photo_blocks" not in _c)
ck("Ultra hint recommends it for style / non-identity work",
   "non-identity" in app._MINIMAX_MODE_HINTS["ultra"])

# --- 6. prefs.json untouched -------------------------------------------------------------------------
_hash1 = hashlib.sha256(open(PREFS, "rb").read()).hexdigest() if os.path.exists(PREFS) else None
ck("prefs.json byte-identical", _hash0 == _hash1)

root.destroy()
print()
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("ALL PASS")

