"""LoKR on the Training tab — headless GUI verification (no GPU, no model load).

Covers the Phase 3 wiring: the Network Type control exists only under Krea 2, LoKR swaps the
rank/alpha rows for the Factor dial, the command builder emits the flags, and both keys ride
the preset/persistence sweep.
"""
import os
import sys

os.environ["FIZGIG_NO_PERSIST"] = "1"
REPO = r"W:/Peter/Documents/Development/Fizgig"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import tkinter as tk  # noqa: E402
import lora_trainer_gui as G  # noqa: E402

G.LAST_USED_FILE = os.path.join(os.environ["TEMP"], "nope", ".last_used.json")

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


def _visible(w):
    try:
        return bool(w.winfo_manager())
    except tk.TclError:
        return False


root = tk.Tk()
root.withdraw()
g = G.LoRATrainerGUI(root)

KREA2 = next(k for k in G.ARCHITECTURES if G.ARCHITECTURES[k].get("is_krea2"))
KLEIN = next(k for k in G.ARCHITECTURES if not G.ARCHITECTURES[k].get("is_krea2"))

# --- 1. widgets + per-family visibility ---------------------------------------------------
ck("NETWORK_TYPE and LOKR_FACTOR widgets exist",
   "NETWORK_TYPE" in g.entries and "LOKR_FACTOR" in g.entries)
ck("  default is standard LoRA (LoKR one pick away)",
   g.entries["NETWORK_TYPE"].get() == "LoRA (standard)")
ck("  default factor is 8", g.entries["LOKR_FACTOR"].get() == "8",
   g.entries["LOKR_FACTOR"].get())

# The combo/entry are packed inside row frames (widget + hint side by side), so the frames
# are what get shown/hidden — check those.
g.architecture_var.set(KLEIN)
g.update_ui_for_architecture()
root.update()
ck("Klein: Network Type hidden", not _visible(g._network_type_rowf))
ck("  Klein: factor hidden, rank/alpha shown",
   not _visible(g._lokr_factor_rowf) and _visible(g.entries["NETWORK_DIM"]))

g.architecture_var.set(KREA2)
g.update_ui_for_architecture()
root.update()
ck("Krea 2: Network Type shown", _visible(g._network_type_rowf))
ck("  default LoRA -> rank/alpha shown, factor (and its hint) hidden",
   _visible(g.entries["NETWORK_DIM"]) and not _visible(g._lokr_factor_rowf))

g.entries["NETWORK_TYPE"].set("LoKR (Kronecker)")
g._on_network_type_changed()
root.update()
ck("LoKR selected -> factor row (entry + sweet-spot hint) shown, rank/alpha hidden",
   _visible(g._lokr_factor_rowf) and not _visible(g.entries["NETWORK_DIM"])
   and not _visible(g.entries["NETWORK_ALPHA"]))

# Switching to Klein with LoKR selected must restore rank/alpha (Klein trains standard only).
g.architecture_var.set(KLEIN)
g.update_ui_for_architecture()
root.update()
ck("Klein with LoKR still selected -> rank/alpha back, factor + combo hidden",
   _visible(g.entries["NETWORK_DIM"]) and not _visible(g._lokr_factor_rowf)
   and not _visible(g._network_type_rowf))
g.architecture_var.set(KREA2)
g.update_ui_for_architecture()
root.update()

# --- 2. command builder -------------------------------------------------------------------
g.settings.update({"NETWORK_TYPE": "LoKR (Kronecker)", "LOKR_FACTOR": 16,
                   "NETWORK_DIM": 8, "NETWORK_ALPHA": 8, "DATASET_CONFIG": "x.toml",
                   "LORA_OUTPUT_DIR": "out", "LORA_NAME": "n", "LEARNING_RATE": 1e-4,
                   "MAX_TRAIN_EPOCHS": 2, "SAVE_EVERY_N_EPOCHS": 1, "BLOCKS_SWAP": 0,
                   "SEED": 42})
cmd = g._build_krea2_train_command()
ck("LoKR -> command carries --network_type lokr --lokr_factor 16",
   "--network_type" in cmd and cmd[cmd.index("--network_type") + 1] == "lokr"
   and cmd[cmd.index("--lokr_factor") + 1] == "16")

g.settings["NETWORK_TYPE"] = "LoRA (standard)"
cmd = g._build_krea2_train_command()
ck("standard LoRA -> no network_type flag at all", "--network_type" not in cmd)

# --- 3. persistence sweep -----------------------------------------------------------------
vals = g._collect_preset_values()
ck("preset sweep captures NETWORK_TYPE + LOKR_FACTOR",
   "NETWORK_TYPE" in vals and "LOKR_FACTOR" in vals,
   {k: vals.get(k) for k in ("NETWORK_TYPE", "LOKR_FACTOR")})
ck("NETWORK_TYPE is strict-combo protected (junk can't be .set() onto it)",
   "NETWORK_TYPE" in G.LoRATrainerGUI._STRICT_COMBO_KEYS)
for name, preset in G.KREA2_BUILT_IN_PRESETS.items():
    ck(f"  built-in '{name[:30]}...' pins standard LoRA",
       preset.get("NETWORK_TYPE") == "LoRA (standard)")

# The hint under Network Type follows the selection (9 Sep 2026).
g.entries["NETWORK_TYPE"].set("LoKR (Kronecker)"); g._on_network_type_changed(); root.update()
ck("LoKR selected -> hint gives the 5e-5 / Adaptive 1e-4-5e-5 learning-rate advice",
   "5e-5" in g._network_type_hint.cget("text") and "Max 1e-4" in g._network_type_hint.cget("text"))
g.entries["NETWORK_TYPE"].set("LoRA (standard)"); g._on_network_type_changed(); root.update()
ck("LoRA selected -> the general trade line", g._network_type_hint.cget("text").startswith("LoKR: drop the learning rate"))

# Applying a built-in preset resets a LoKR selection back to standard.
g.entries["NETWORK_TYPE"].set("LoKR (Kronecker)")
g._apply_preset_values(next(iter(G.KREA2_BUILT_IN_PRESETS.values())))
root.update()
ck("loading a built-in preset resets Network Type to standard",
   g.entries["NETWORK_TYPE"].get() == "LoRA (standard)", g.entries["NETWORK_TYPE"].get())

# --- 4. fine-tune × LoKR mutual exclusion (ft-rotation branch) ----------------------------
# FT trains the base; the adapter is inert. LoKR must vanish from the UI and the command.
if hasattr(g, "krea2_finetune_var"):
    g.entries["NETWORK_TYPE"].set("LoKR (Kronecker)")
    g.krea2_finetune_var.set(True)
    g._apply_krea2_ft_visibility()
    root.update()
    ck("FT on -> Network Type row hidden", not _visible(g._network_type_rowf))
    ck("  FT on -> factor hidden, rank/alpha shown (FT recipe uses them for nothing, "
       "but Klein parity keeps them)", not _visible(g._lokr_factor_rowf)
       and _visible(g.entries["NETWORK_DIM"]))
    g.settings["NETWORK_TYPE"] = "LoKR (Kronecker)"
    cmd = g._build_krea2_train_command()
    ck("  FT on -> --network_type NOT emitted even with LoKR selected",
       "--network_type" not in cmd)
    ck("  FT on -> FT flags emitted", "--finetune_rotation" in cmd)
    ck("  FT recipe pins standard LoRA",
       g.KREA2_FT_DEFAULTS.get("NETWORK_TYPE") == "LoRA (standard)")
    g.krea2_finetune_var.set(False)
    g._apply_krea2_ft_visibility()
    root.update()
    ck("FT off -> Network Type row back, LoKR wiring restored",
       _visible(g._network_type_rowf) and _visible(g._lokr_factor_rowf))
    cmd = g._build_krea2_train_command()
    ck("  FT off -> --network_type lokr emitted again",
       "--network_type" in cmd and "--finetune_rotation" not in cmd)

root.destroy()
print()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
