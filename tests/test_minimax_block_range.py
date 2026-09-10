r"""MiniMax H3 "Blocks to Train" — the block-selection experiment knob, end to end.

The whole point of the knob is A/B: train blocks 14-37, compare against a full-model run. That
only means anything if the range is exactly what got trained, so this pins the module set rather
than the flag string — a range that silently trained 50 blocks (or 0) would produce a null
result that reads as "the hypothesis is wrong".

Traps this exists to catch:
  * `token_refiner.blocks.N` also matches a `blocks\.\d+` pattern — the refiner must be trained
    in EVERY range (holding it constant is what keeps two ranges comparable), and must never be
    counted as a DiT block;
  * the pruned checkpoint adds AdaLN to the pattern set, so narrowing must apply to that too
    rather than leaving AdaLN training on all 50 blocks;
  * block 4 must not be matched by the alternation for a 14-37 range (a naive `1|4|...` join).

Run: venv/Scripts/python.exe tests/test_minimax_block_range.py
"""
import os
import re
import sys

os.environ["FIZGIG_NO_PERSIST"] = "1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402
from fizgig.minimax.model import MiniMaxH3Config, MiniMaxH3DiT  # noqa: E402
from fizgig.minimax.trainer import (DEFAULT_INCLUDE_PATTERNS,  # noqa: E402
                                    PRUNED_INCLUDE_PATTERNS, format_block_spec,
                                    parse_block_spec, restrict_patterns_to_blocks)
from fizgig.networks.lora import create_network  # noqa: E402

FAILS = []


def ck(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


CFG = dict(hidden_size=64, num_layers=20, token_refiner_num_layers=2, num_attention_heads=4,
           attention_head_dim=16, ffn_hidden_size=48, latents_dim=24, audio_latents_dim=6,
           patch_size=(1, 2, 2), text_dim=32, timestep_input_dim=16, time_embed_hidden_size=64,
           time_embed_dim=32, rope_inv_freq_len=2)
BLOCK_RE = re.compile(r"blocks_(\d+)_")


def wrap(patterns):
    """-> (set of DiT block indices trained, count of token_refiner modules trained)"""
    torch.manual_seed(0)
    dit = MiniMaxH3DiT(MiniMaxH3Config(**CFG))
    dit.requires_grad_(False)
    net = create_network(None, "lora_unet", 1.0, 4, 4, None, [], dit, include_patterns=patterns)
    blocks, refiner = set(), 0
    for lora in net.unet_loras:
        n = lora.lora_name
        if "token_refiner" in n:
            refiner += 1
            continue
        m = BLOCK_RE.search(n)
        if m:
            blocks.add(int(m.group(1)))
    return blocks, refiner


# --- 0. the spec parser: ranges, singles, mixed --------------------------------------------------
ck("a plain range", parse_block_spec("14-17") == [14, 15, 16, 17])
ck("a single block", parse_block_spec("22") == [22])
ck("the documented mixed example",
   parse_block_spec("3-12, 14-15, 22,27,31-33")
   == [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 22, 27, 31, 32, 33])
ck("whitespace anywhere", parse_block_spec("  3 - 5 ,  9 ") == [3, 4, 5, 9])
ck("overlaps collapse rather than double-count", parse_block_spec("3-6,5-8") == [3, 4, 5, 6, 7, 8])
ck("out of order is sorted", parse_block_spec("30,2,10-11") == [2, 10, 11, 30])
ck("a trailing comma is tolerated", parse_block_spec("1-3,") == [1, 2, 3])
try:
    parse_block_spec("48-52", 50)
    _ok = False
except ValueError as e:
    _ok = "do not exist" in str(e)
ck("an out-of-range block is refused, not silently dropped", _ok)
ck("in-range passes the bounds check", parse_block_spec("48-49", 50) == [48, 49])

ck("format: runs collapse", format_block_spec([3, 4, 5, 7]) == "3-5,7", format_block_spec([3, 4, 5, 7]))
ck("format: round-trips the mixed example",
   format_block_spec(parse_block_spec("3-12, 14-15, 22,27,31-33")) == "3-12,14-15,22,27,31-33",
   format_block_spec(parse_block_spec("3-12, 14-15, 22,27,31-33")))

# --- 1. the range is exactly the range ---------------------------------------------------------
full_blocks, full_refiner = wrap(DEFAULT_INCLUDE_PATTERNS)
ck("baseline trains every block", full_blocks == set(range(20)), sorted(full_blocks)[:3])
ck("baseline trains the text refiner", full_refiner > 0, full_refiner)

for lo, hi in ((5, 12), (0, 0), (14, 19), (10, 19)):
    got, ref = wrap(restrict_patterns_to_blocks(DEFAULT_INCLUDE_PATTERNS, f"{lo}-{hi}"))
    ck(f"range {lo}-{hi} trains exactly those blocks", got == set(range(lo, hi + 1)),
       f"got {sorted(got)}")
    ck(f"range {lo}-{hi} still trains the text refiner", ref == full_refiner, ref)

# The alternation trap: a 14-19 range must not match block 1 or 4 via a sloppy join.
got, _ = wrap(restrict_patterns_to_blocks(DEFAULT_INCLUDE_PATTERNS, "14-19"))
ck("single-digit blocks are NOT caught by a two-digit range",
   not ({1, 4, 9} & got), f"leaked {sorted({1, 4, 9} & got)}")

# --- 2. per-module-kind coverage inside the range -----------------------------------------------
def kinds(patterns):
    torch.manual_seed(0)
    dit = MiniMaxH3DiT(MiniMaxH3Config(**CFG))
    dit.requires_grad_(False)
    net = create_network(None, "lora_unet", 1.0, 4, 4, None, [], dit, include_patterns=patterns)
    # DiT blocks only: the refiner has its own attn/mlp modules and is never narrowed, so
    # counting it here would hide a 2x relationship behind a constant (44 vs 24, not 40 vs 20).
    return {k: sum(1 for lo in net.unet_loras
                   if f"_{k}_" in lo.lora_name and "token_refiner" not in lo.lora_name)
            for k in ("attn", "mlp", "adaln")}


k_full = kinds(DEFAULT_INCLUDE_PATTERNS)
k_half = kinds(restrict_patterns_to_blocks(DEFAULT_INCLUDE_PATTERNS, "10-19"))
ck("attention modules halve with half the blocks", k_half["attn"] * 2 == k_full["attn"],
   f"{k_half['attn']} vs {k_full['attn']}")
ck("MLP modules halve with half the blocks", k_half["mlp"] * 2 == k_full["mlp"],
   f"{k_half['mlp']} vs {k_full['mlp']}")

# --- 3. the pruned pattern set (AdaLN) narrows too ----------------------------------------------
p_full = kinds(PRUNED_INCLUDE_PATTERNS)
p_half = kinds(restrict_patterns_to_blocks(PRUNED_INCLUDE_PATTERNS, "10-19"))
ck("the pruned set trains AdaLN at all", p_full["adaln"] > 0, p_full["adaln"])
ck("AdaLN narrows with the range too (not left on all 50)",
   p_half["adaln"] * 2 == p_full["adaln"], f"{p_half['adaln']} vs {p_full['adaln']}")

# --- 3b. a discontiguous selection wraps exactly those blocks -----------------------------------
got, ref = wrap(restrict_patterns_to_blocks(DEFAULT_INCLUDE_PATTERNS, "1-3, 7, 11-12"))
ck("a mixed spec trains exactly those blocks", got == {1, 2, 3, 7, 11, 12}, sorted(got))
ck("a mixed spec still trains the text refiner", ref == full_refiner, ref)
p_mixed = kinds(restrict_patterns_to_blocks(PRUNED_INCLUDE_PATTERNS, "1-3, 7, 11-12"))
ck("AdaLN follows a mixed spec too", p_mixed["adaln"] * (20 / 6) == p_full["adaln"],
   f"{p_mixed['adaln']} for 6 blocks vs {p_full['adaln']} for 20")

# --- 3c. Train AdaLN toggle: the pattern set, not just the flag ---------------------------------
# AdaLN reads the timestep and nothing else (DiTBlock.forward -> adaln_proj(t_emb)), so it cannot
# encode identity — but it carries ~45% of weight movement on the pruned checkpoint. The toggle
# must remove exactly the AdaLN modules and touch nothing else.
_no_adaln = [p for p in PRUNED_INCLUDE_PATTERNS if "adaln" not in p]
k_no = kinds(_no_adaln)
ck("AdaLN off removes every AdaLN module", k_no["adaln"] == 0, k_no["adaln"])
ck("AdaLN off leaves attention untouched", k_no["attn"] == p_full["attn"],
   f"{k_no['attn']} vs {p_full['attn']}")
ck("AdaLN off leaves the MLPs untouched", k_no["mlp"] == p_full["mlp"],
   f"{k_no['mlp']} vs {p_full['mlp']}")
_, _ref_no = wrap(_no_adaln)
ck("AdaLN off leaves the text refiner trained", _ref_no == full_refiner, _ref_no)
ck("the bf16 pattern set has no AdaLN to remove in the first place",
   [p for p in DEFAULT_INCLUDE_PATTERNS if "adaln" not in p] == DEFAULT_INCLUDE_PATTERNS)
# ...and it composes with a block range
_combo = restrict_patterns_to_blocks(_no_adaln, "10-19")
k_combo = kinds(_combo)
ck("AdaLN off composes with a block range", k_combo["adaln"] == 0 and k_combo["attn"] * 2 == p_full["attn"],
   f"adaln {k_combo['adaln']}, attn {k_combo['attn']} vs {p_full['attn']}")

# --- 4. bad input is refused, not silently ignored ----------------------------------------------
for bad in ("", "abc", "37-14", "14..37", "14-", None, "3-12; 14", "1-2-3", "-5", "3,,x"):
    try:
        restrict_patterns_to_blocks(DEFAULT_INCLUDE_PATTERNS, bad)
        ok = False
    except ValueError:
        ok = True
    ck(f"refuses {bad!r}", ok)

# --- 5. GUI: the dropdown, the flag, and the queue row -------------------------------------------
import tkinter as tk  # noqa: E402

import lora_trainer_gui as G  # noqa: E402

G.LAST_USED_FILE = os.path.join(os.environ.get("TEMP", "/tmp"), "nope", ".last_used.json")
G.LoRATrainerGUI.save_prefs = lambda self, *a, **k: None
G.LoRATrainerGUI._save_training_queue = lambda self, *a, **k: None

for opt in G.MINIMAX_BLOCK_OPTIONS:
    tok = G.minimax_block_spec(opt)
    try:
        ok = tok == "all" or bool(parse_block_spec(tok, G.MINIMAX_NUM_BLOCKS))
    except ValueError:
        ok = False
    ck(f"preset '{tok}' is a spec the trainer accepts", ok, opt)

# The separator trap: labels use "·" precisely because "-" and " " occur inside a spec.
ck("a typed spec survives extraction intact",
   G.minimax_block_spec("3-12, 14-15, 22,27,31-33") == "3-12, 14-15, 22,27,31-33")
ck("a label is reduced to its spec", G.minimax_block_spec("14-37 · middle band") == "14-37")
ck("an empty box means all", G.minimax_block_spec("") == "all")
ck("no preset label contains a hyphen-space-hyphen that would split a spec",
   all("·" in o for o in G.MINIMAX_BLOCK_OPTIONS))

root = tk.Tk()
root.withdraw()
app = G.LoRATrainerGUI(root)
app.architecture_var.set("MiniMax H3 (experimental)")
# Validation now checks model-path EXISTENCE before anything else and returns early on a
# missing file — so the block-spec checks below need real (empty) files to get past it.
import tempfile  # noqa: E402

_vd = tempfile.mkdtemp(prefix="fizgig-blockrange-")
_fake = {}
for _n in ("dit", "vae", "te", "ds.toml"):
    _fp = os.path.join(_vd, _n if "." in _n else _n + ".safetensors")
    open(_fp, "wb").close()
    _fake[_n] = _fp
app.prefs_vars["minimax_dit"].set(_fake["dit"])
app.prefs_vars["minimax_vae"].set(_fake["vae"])
app.prefs_vars["minimax_text_encoder"].set(_fake["te"])
BASE = {
    "DATASET_CONFIG": _fake["ds.toml"], "LORA_OUTPUT_DIR": _vd, "LORA_NAME": "t",
    "NETWORK_DIM": 16, "NETWORK_ALPHA": 16, "LEARNING_RATE": 1e-4, "MAX_TRAIN_EPOCHS": 20,
    "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42, "BLOCKS_SWAP": "auto", "ADAPTIVE_LR": False,
    "MAX_GRAD_NORM": "1.0", "OPTIMIZER_TYPE": "adamw8bit", "OPTIMIZER_ARGS": "",
    "METADATA_TITLE": "", "METADATA_AUTHOR": "", "METADATA_DESCRIPTION": "",
    "METADATA_LICENSE": "", "METADATA_TAGS": "", "METADATA_TRIGGER_PHRASE": "",
}
CFGA = G.ARCHITECTURES["MiniMax H3 (experimental)"]


def blocks_flag(value):
    app.settings = dict(BASE, MINIMAX_BLOCKS=value)
    cmd = [str(x) for x in app.build_training_command(CFGA)]
    return cmd[cmd.index("--train_blocks") + 1] if "--train_blocks" in cmd else None


ck("'all' sends no flag (the trainer's own default)", blocks_flag("all") is None)
ck("a range is sent", blocks_flag("14-37") == "14-37")
ck("a labelled option is reduced to its spec",
   blocks_flag("14-37 · middle band") == "14-37")
ck("a hand-typed mixed spec reaches the CLI WHOLE",
   blocks_flag("3-12, 14-15, 22,27,31-33") == "3-12, 14-15, 22,27,31-33")
ck("an empty setting behaves like 'all'", blocks_flag("") is None)

# ...and the launch check refuses a typo before anything loads. validate_inputs returns a bool
# and reports through a modal, so capture the dialog rather than calling it blind (a real
# messagebox would hang a headless run). Likeness mode deliberately ignores (and disables)
# the box — a stale typo must not block a likeness launch — so untick it for these checks.
app.entries["MINIMAX_LIKENESS_MODE"].set(G.MINIMAX_MODE_OFF)
_shown = []
_real_showerror = G.messagebox.showerror
G.messagebox.showerror = lambda title, msg, **k: _shown.append(msg)
try:
    app.entries["MINIMAX_BLOCKS"].set("3-12, banana")
    app.validate_inputs()
    ck("a bad spec is caught at launch, not after the 21 GB load",
       any("Blocks to Train" in m for m in _shown),
       (_shown[-1].splitlines()[-1] if _shown else "no dialog raised"))

    _shown.clear()
    app.entries["MINIMAX_BLOCKS"].set("48-52")
    app.validate_inputs()
    ck("an out-of-range block is caught at launch too",
       any("Blocks to Train" in m and "do not exist" in m for m in _shown),
       (_shown[-1].splitlines()[-1] if _shown else "no dialog raised"))

    _shown.clear()
    app.entries["MINIMAX_BLOCKS"].set("3-12, 14-15, 22,27,31-33")
    app.validate_inputs()
    ck("a good spec raises no complaint about blocks",
       not any("Blocks to Train" in m for m in _shown))
finally:
    G.messagebox.showerror = _real_showerror
app.entries["MINIMAX_BLOCKS"].set("all · every block (50 of 50)")

# the live readout
for spec, want in (("all · every block (50 of 50)", "all 50"), ("1-3, 7", "✓ 4 of 50"),
                   ("3-12, banana", "✗")):
    app.entries["MINIMAX_BLOCKS"].set(spec)
    app._refresh_minimax_blocks_count()
    ck(f"readout for {spec!r}", want in app._minimax_blocks_count.cget("text"),
       app._minimax_blocks_count.cget("text"))

# --- 5b. the AdaLN toggle through the GUI --------------------------------------------------------
# Reset the blocks box first: the validation checks above deliberately left a bad spec in it, and
# a queue snapshot taken now would carry it into the rows asserted below.
app._select_combo_by_token(app.entries["MINIMAX_BLOCKS"], "all")


def adaln_flag(on):
    app.settings = dict(BASE, MINIMAX_BLOCKS="all", MINIMAX_TRAIN_ADALN=on)
    return "--no_train_adaln" in [str(x) for x in app.build_training_command(CFGA)]


# AdaLN training is RETIRED (9 Aug): the pruned builds everyone deploys on cannot load AdaLN
# LoRA keys, so training it only wastes capacity. The checkbox is hidden and the builder sends
# the opt-out UNCONDITIONALLY - a stale saved setting must not be able to switch it back on.
ck("AdaLN opt-out is sent even when the setting says ON", adaln_flag(True))
ck("...and when it says off", adaln_flag(False))
ck("...and when the setting is missing entirely", "--no_train_adaln" in [
    str(x) for x in (app.__setattr__("settings", dict(BASE, MINIMAX_BLOCKS="all")) or
                     app.build_training_command(CFGA))])

app.entries["MINIMAX_TRAIN_ADALN"].set(False)
_snap_adaln = app._collect_preset_values()
ck("the toggle is captured in a snapshot", _snap_adaln.get("MINIMAX_TRAIN_ADALN") is False,
   _snap_adaln.get("MINIMAX_TRAIN_ADALN"))
_item_adaln = app._queue_snapshot()
app.entries["MINIMAX_TRAIN_ADALN"].set(True)
app._apply_queue_item(_item_adaln)
ck("a queued run replays with AdaLN off",
   app.entries["MINIMAX_TRAIN_ADALN"].get() is False, app.entries["MINIMAX_TRAIN_ADALN"].get())
ck("the queue manager shows it", "no adaln" in app._queue_row_summary(_item_adaln)[1],
   app._queue_row_summary(_item_adaln)[1].replace("\n", " | "))
app.entries["MINIMAX_TRAIN_ADALN"].set(True)
ck("the queue manager stays quiet when AdaLN is on",
   "adaln" not in app._queue_row_summary(app._queue_snapshot())[1])

# (Likeness is still unticked from the validation section — a manual range is a
# likeness-off workflow now.)
app._select_combo_by_token(app.entries["MINIMAX_BLOCKS"], "25-49")
snap = app._collect_preset_values()
item = app._queue_snapshot()
app._select_combo_by_token(app.entries["MINIMAX_BLOCKS"], "all")
app._apply_queue_item(item)
ck("a queued run replays with its own block range",
   app.entries["MINIMAX_BLOCKS"].get().split(" ")[0] == "25-49",
   app.entries["MINIMAX_BLOCKS"].get())
ck("the queue manager shows the range", "blocks 25-49" in app._queue_row_summary(item)[1],
   app._queue_row_summary(item)[1].replace("\n", " | "))
_all_item = dict(item, preset=dict(snap, MINIMAX_BLOCKS="all"))
ck("the queue manager stays quiet for a full-model run",
   "blocks" not in app._queue_row_summary(_all_item)[1])

# The AdaLN widgets are retired and hidden under EVERY family, so they are not part of the
# "shown for MiniMax" set any more - only the Blocks to Train row is.
WIDGETS = ("_minimax_blocks_label", "_minimax_blocks_frame", "_minimax_blocks_hint")
app.update_ui_for_architecture()
ck("shown for MiniMax", all(bool(getattr(app, n).winfo_manager()) for n in WIDGETS))
for arch in (a for a in G.ARCHITECTURES if not G.ARCHITECTURES[a].get("is_minimax")):
    app.architecture_var.set(arch)
    app.update_ui_for_architecture()
    ck(f"hidden for {arch}", not any(bool(getattr(app, n).winfo_manager()) for n in WIDGETS))

root.destroy()
print()
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
