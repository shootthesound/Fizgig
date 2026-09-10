"""MiniMax H3 GUI integration (Stage 2) — headless, FIZGIG_NO_PERSIST so nothing touches
the real last_used.json / prefs.

Pins:
  * the arch entry exists and is selectable, is_minimax / is_krea2 resolve correctly;
  * switching to MiniMax (and cycling all three families) raises nothing;
  * the built-in presets are the MiniMax set;
  * the three command builders dispatch to the minimax_* scripts with the right flags,
    including adaptive LR;
  * validate_inputs requires the three minimax_* Preferences paths.

Run: venv/Scripts/python.exe tests/test_minimax_gui.py
"""

import os
import sys

os.environ["FIZGIG_NO_PERSIST"] = "1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import tkinter as tk  # noqa: E402
import lora_trainer_gui as G  # noqa: E402

G.LAST_USED_FILE = os.path.join(os.environ.get("TEMP", "/tmp"), "nope", ".last_used.json")
# Belt and braces: neuter the prefs WRITER outright for this whole process. FIZGIG_NO_PERSIST
# already guards it, but this test has to defeat that guard briefly (below) to exercise the
# last-train save — and a global guard flipped off is exactly how a test clobbers the user's
# real prefs.json. Never let the writer be reachable at all.
G.save_prefs = lambda *_a, **_k: None

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


ARCH = "MiniMax H3"
# The label lost its "(experimental)" in 3.6.1. Saved configs still carry the old one, and
# every ARCHITECTURES lookup is a .get() defaulting to Klein - so a missed alias does not
# raise, it silently returns a returning user to the wrong model family.
ARCH_OLD = "MiniMax H3 (experimental)"

root = tk.Tk()
root.withdraw()
app = G.LoRATrainerGUI(root)

# --- arch entry present + flags ----------------------------------------------------------
ck("MiniMax arch is in ARCHITECTURES + list", ARCH in G.ARCHITECTURES and ARCH in G.ARCHITECTURE_LIST)
cfg = G.ARCHITECTURES[ARCH]
ck("config: is_minimax, samples enabled, native scripts",
   cfg.get("is_minimax") and cfg.get("supports_samples") is True
   and "minimax_train.py" in cfg["train_script"])

# --- cycling all three families raises nothing (regression guard) ------------------------
try:
    for a in ["Flux 2 Klein Base 9B", "Krea 2", ARCH, "Krea 2", "Flux 2 Klein Base 9B", ARCH]:
        app.architecture_var.set(a)
        app.on_architecture_changed()
    cycled = True
except Exception as e:  # noqa: BLE001
    cycled = False
    print("  cycle error:", e)
ck("cycling Klein/Krea2/MiniMax raises nothing", cycled)
ck("MiniMax selected -> is_minimax True, is_krea2 False",
   app._is_minimax_arch() and not app._is_krea2_arch())

# --- built-in presets are the MiniMax set ------------------------------------------------
builtins = list(app._builtins_for_arch(ARCH).keys())
ck("MiniMax built-in presets load", builtins and all("MiniMax" in b for b in builtins), builtins)

# The preset ships adamw, NOT adamw8bit (2026-08-06): full-precision optimizer state was the
# single biggest likeness change measured on H3, after every other knob had been swept with
# likeness stuck near 40-50%. OPTIMIZER_TYPE is a _STRICT_COMBO_KEY, so a name the dropdown
# does not offer is silently DROPPED rather than set — the preset would then quietly train on
# whatever was already selected. Hence both halves: the value, and that it actually applies.
_p = app._builtins_for_arch(ARCH)[builtins[0]]
ck("the MiniMax preset specifies adamw", _p.get("OPTIMIZER_TYPE") == "adamw",
   _p.get("OPTIMIZER_TYPE"))
_combo = app.entries.get("OPTIMIZER_TYPE")
ck("...and the MiniMax optimizer dropdown actually offers it",
   "adamw" in list(_combo["values"]), list(_combo["values"]))
_combo.set("adamw8bit")
app._apply_preset_values(_p)
ck("...so loading the preset really selects adamw", _combo.get() == "adamw", _combo.get())

# Noise schedule: mid-concentrated (MINIMAX_LOGNORM) is RETIRED — the builder deliberately
# ignores a stale saved key rather than migrating it, and only the plain uniform-base shift
# ships. Pinned so the retirement can't half-return: no lognorm var, no lognorm converter,
# and the low-noise percentage still applies from a preset.
ck("the preset asks for 60% low-noise", _p.get("MINIMAX_LOWNOISE_PCT") == "60",
   _p.get("MINIMAX_LOWNOISE_PCT"))
ck("mid-concentrated is fully retired (no var, no converter)",
   not hasattr(app, "minimax_lognorm_var")
   and not hasattr(G, "minimax_lownoise_to_lognorm_shift"))
app.entries["MINIMAX_LOWNOISE_PCT"].delete(0, "end")
app.entries["MINIMAX_LOWNOISE_PCT"].insert(0, "22")
app._apply_preset_values(_p)
ck("...the percentage actually applies to the UI",
   app.entries["MINIMAX_LOWNOISE_PCT"].get() == "60",
   app.entries["MINIMAX_LOWNOISE_PCT"].get())
_sh_uni = G.minimax_lownoise_to_shift("60")
ck("60% resolves to the uniform-base shift ~0.667", abs(_sh_uni - 0.667) < 0.01, _sh_uni)

# The rest of the shipped baseline. Pinned as a set because the point of this preset is that it
# is the PLAIN configuration every A/B is measured against — one experiment knob left on by
# accident silently contaminates every comparison run from it, and none of these show up as an
# error.
# The FIRST preset is Fast (Peter, 22 Aug): rank 8, 50 epochs, flat 2e-4, ramp Off — what a
# family switch and a fresh start apply.
for _k, _want in (("NETWORK_DIM", 8), ("NETWORK_ALPHA", 8), ("MAX_TRAIN_EPOCHS", 50),
                  ("DATASET_MEGAPIXELS", "0.25"), ("MINIMAX_BLOCKS", "all"),
                  ("MINIMAX_TRAIN_ADALN", False), ("MINIMAX_TRAIN_REFINER", False),
                  ("MINIMAX_SLOW_BLOCKS", ""),
                  ("MINIMAX_DISTILL", False), ("ADAPTIVE_LR", False),
                  ("NETWORK_TYPE", "LoRA (standard)"), ("LOKR_FACTOR", 8),
                  ("MINIMAX_ADAPTER_RAMP", "Off")):
    ck(f"preset {_k} == {_want!r}", _p.get(_k) == _want, _p.get(_k))

# AdaLN and Distill are booleans held in DIFFERENT places — MINIMAX_TRAIN_ADALN is a BooleanVar
# inside self.entries (generic loop), MINIMAX_DISTILL is a bare attribute needing its own branch
# in _apply_preset_values. A preset key with no matching branch is dropped in silence, so assert
# the applied state rather than the dict.
app.entries["MINIMAX_TRAIN_ADALN"].set(True)
app.minimax_distill_var.set(True)
app._apply_preset_values(_p)
ck("AdaLN training really switches OFF when the preset loads",
   app.entries["MINIMAX_TRAIN_ADALN"].get() is False, app.entries["MINIMAX_TRAIN_ADALN"].get())
ck("reference distillation really switches OFF too",
   app.minimax_distill_var.get() is False, app.minimax_distill_var.get())

# Three ship (Fast leads since 22 Aug; the rank-16 Lower-LR recipe is one dropdown away;
# Style is the block-spec preset). Fast is built by SPREADING the Lower-LR defaults, so the
# pin that matters is that the two differ in EXACTLY the intended fields; anything else
# means they have started to drift apart, which is what retired the previous Fast preset.
ck("three MiniMax built-in presets ship", len(builtins) == 3, builtins)
ck("Fast leads the list (family-switch default)", "Fast" in builtins[0], builtins[0])
_slow = app._builtins_for_arch(ARCH)[builtins[1]]
ck("the second is the Lower-LR rank-16 recipe", "Lower LR" in builtins[1], builtins[1])
_diff = {k for k in set(_p) | set(_slow) if _p.get(k) != _slow.get(k)}
ck("Fast differs from Lower-LR in exactly the four intended fields",
   _diff == {"NETWORK_DIM", "NETWORK_ALPHA", "MAX_TRAIN_EPOCHS", "LEARNING_RATE"},
   sorted(_diff))
for _k, _want in (("NETWORK_DIM", 16), ("NETWORK_ALPHA", 16), ("MAX_TRAIN_EPOCHS", 60),
                  ("MINIMAX_ADAPTER_RAMP", "Off"), ("LEARNING_RATE", 1e-4),
                  ("ADAPTIVE_LR", False)):
    ck(f"Lower-LR preset {_k} == {_want!r}", _slow.get(_k) == _want, _slow.get(_k))
ck("Fast preset LR is the 2e-4 ceiling", _p.get("LEARNING_RATE") == 2e-4)
for _k in ("MINIMAX_BLOCK_LIMIT", "MINIMAX_LR_WARMUP"):
    ck(f"preset {_k} is Off", str(_p.get(_k, "")).split(" ")[0] == "Off", _p.get(_k))
ck("preset MINIMAX_EMA is 0.98 (on by default, 9 Sep A/B)", str(_p.get("MINIMAX_EMA", "")).split(" ")[0] == "0.98", _p.get("MINIMAX_EMA"))
app._apply_preset_values(_p)                      # leave the Defaults preset loaded

# --- command builders --------------------------------------------------------------------
app.prefs_vars["minimax_dit"].set("X:/dit.safetensors")
app.prefs_vars["minimax_vae"].set("X:/vae.safetensors")
app.prefs_vars["minimax_text_encoder"].set("X:/te.safetensors")
app.settings = {
    "DATASET_CONFIG": "X:/ds.toml", "LORA_OUTPUT_DIR": "X:/out", "LORA_NAME": "mytest",
    "NETWORK_DIM": 16, "NETWORK_ALPHA": 16, "LEARNING_RATE": 1e-4, "MAX_TRAIN_EPOCHS": 20,
    "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42, "ADAPTIVE_LR": True,
    "ADAPTIVE_LR_MIN": "5e-5", "ADAPTIVE_LR_MAX": "4e-4", "MAX_GRAD_NORM": "1.0",
    "OPTIMIZER_TYPE": "adamw8bit", "OPTIMIZER_ARGS": "",
    "METADATA_TITLE": "", "METADATA_AUTHOR": "", "METADATA_DESCRIPTION": "",
    "METADATA_LICENSE": "", "METADATA_TAGS": "", "METADATA_TRIGGER_PHRASE": "",
}
tc = [str(x) for x in app.build_training_command(cfg)]
cl = [str(x) for x in app.build_cache_latents_command(cfg)]
ct = [str(x) for x in app.build_cache_text_command(cfg)]
ck("train cmd -> minimax_train.py + --dit", any("minimax_train.py" in x for x in tc)
   and "--dit" in tc)
# The 9 Aug locks: the settings dict above deliberately carries the RETIRED values
# (ADAPTIVE_LR True, adamw8bit) — the builder must ignore/override them, or a user's stale
# saved settings silently dismantle the stability stack.
ck("adaptive LR is locked OUT even when the saved setting says on",
   "--adaptive_lr" not in tc)
ck("optimizer is locked to adamw even when the saved setting says adamw8bit",
   tc[tc.index("--optimizer_type") + 1] == "adamw")
ck("AdaLN is locked off (always opts out)", "--no_train_adaln" in tc)
ck("depth-split LR is never emitted", "--slow_blocks" not in tc)
ck("cache-latents cmd -> minimax_cache_latents.py + --vae from prefs",
   any("minimax_cache_latents.py" in x for x in cl) and "X:/vae.safetensors" in cl)
ck("cache-text cmd -> minimax_cache_text.py + --text_encoder from prefs",
   any("minimax_cache_text.py" in x for x in ct) and "X:/te.safetensors" in ct)

# --- Samples tab drives previews: prompts + flags reach the trainer ------------------------
# The whole point of the Samples tab for MiniMax: what the user types must arrive as a prompt
# file, and the 32B TE path must ride along (the trainer needs it to pre-encode).
app.architecture_var.set(ARCH)
app.on_architecture_changed()
ck("Samples tab shows settings (not the 'video model' warning)",
   bool(app.sample_settings_frame.winfo_manager())
   and not app.video_model_warning_frame.winfo_manager())

app.prefs_vars["minimax_text_encoder"].set("X:/te.safetensors")
app.prefs_vars["minimax_vae"].set("X:/vae.safetensors")
app.sample_enabled_var.set(True)
app.sample_prompt_text.delete("1.0", tk.END)
app.sample_prompt_text.insert("1.0", "a portrait of zwxem\n# a comment line\na wide shot of zwxem")
app.sample_every_n_epochs_var.set("1")
app.sample_width_var.set("512")
app.sample_height_var.set("512")
app.sample_steps_var.set("8")
app.sample_seed_var.set("42")
app.settings["LORA_OUTPUT_DIR"] = os.path.join(os.environ.get("TEMP", "/tmp"), "mm_gui_test")
cmd = [str(x) for x in app.build_training_command(cfg)]
ck("sample flags reach the trainer (prompts/cadence/size/steps/TE)",
   all(f in cmd for f in ("--sample_prompts", "--sample_every_n_epochs", "--sample_width",
                          "--sample_steps", "--text_encoder")),
   [f for f in ("--sample_prompts", "--sample_every_n_epochs", "--sample_width",
                "--sample_steps", "--text_encoder") if f not in cmd])
# Retired controls must never reach the trainer, whatever a preset or restored config holds.
# This fails SILENTLY otherwise: the flag is accepted, the run just quietly trains with a
# movement cap or a warmup ramp nobody asked for and no visible control to explain it.
# NOTE the builder reads app.settings, NOT the dict handed to build_training_command — so the
# stale value has to be planted there or the check is vacuous.
for _k, _val, _flag in (("MINIMAX_BLOCK_LIMIT", "1.25 x median (default)", "--block_limit"),
                        ("MINIMAX_LR_WARMUP", "2 epochs", "--lr_warmup_epochs")):
    _was = app.settings.get(_k)
    app.settings[_k] = _val
    _c = [str(x) for x in app.build_training_command(cfg)]
    ck(f"a stale {_k} does not emit {_flag}", _flag not in _c,
       [x for x in _c if x == _flag])
    app.settings[_k] = _was
_was_ema = app.settings.get("MINIMAX_EMA")
app.settings["MINIMAX_EMA"] = "0.99 (stronger)"
ck("EMA still reaches the trainer when set",
   "--ema_decay" in [str(x) for x in app.build_training_command(cfg)])
app.settings["MINIMAX_EMA"] = _was_ema

_pf = cmd[cmd.index("--sample_prompts") + 1]
_lines = open(_pf, encoding="utf-8").read().strip().splitlines()
ck("the user's prompts are written verbatim, comments stripped",
   _lines == ["a portrait of zwxem", "a wide shot of zwxem"], _lines)
ck("prompt file is MiniMax-named, not a krea2_ artefact",
   os.path.basename(_pf) == "minimax_prompts.txt", os.path.basename(_pf))

# previews off -> no sample flags at all
app.sample_enabled_var.set(False)
cmd_off = [str(x) for x in app.build_training_command(cfg)]
ck("previews disabled sends no sample flags", "--sample_prompts" not in cmd_off)
app.sample_enabled_var.set(True)

# --- Blocks to Train: unticking likeness fills in the recommendation ---------------------
# Peter, 10 Sep 2026: 6-49 beat both the 20-49 window and the full 50 (blocks 0-5 deform anatomy
# and add micro-distortion to audio). Untick has to hand back 6-49, not the do-nothing "all" —
# and must never overwrite a spec the user chose. Silent either way if it regresses.
app.entries["MINIMAX_LIKENESS_OPT"].set(True)
app.entries["MINIMAX_BLOCKS"].config(state="")
app.entries["MINIMAX_BLOCKS"].set("all")
app.entries["MINIMAX_LIKENESS_OPT"].set(False)
ck("unticking Optimised Likeness fills Blocks to Train with 6-49",
   G.minimax_block_spec(app.entries["MINIMAX_BLOCKS"].get()) == G.MINIMAX_FULL_MODEL_BLOCKS,
   app.entries["MINIMAX_BLOCKS"].get())
ck("...and the box is editable again", str(app.entries["MINIMAX_BLOCKS"].cget("state")) != "disabled",
   app.entries["MINIMAX_BLOCKS"].cget("state"))
app.entries["MINIMAX_BLOCKS"].set("14-37 · middle band")
app.entries["MINIMAX_LIKENESS_OPT"].set(True)
app.entries["MINIMAX_LIKENESS_OPT"].set(False)
ck("a chosen spec survives a likeness toggle round-trip",
   G.minimax_block_spec(app.entries["MINIMAX_BLOCKS"].get()) == "14-37",
   app.entries["MINIMAX_BLOCKS"].get())
ck("6-49 is an offered option", any(str(o).split(" ")[0] == "6-49" for o in G.MINIMAX_BLOCK_OPTIONS),
   G.MINIMAX_BLOCK_OPTIONS)
app.entries["MINIMAX_LIKENESS_OPT"].set(True)

# --- validate_inputs requires the three minimax_* paths ----------------------------------
# validate_inputs pops a modal messagebox on failure (blocks headless) and returns False —
# stub showerror to capture the message text instead.
app.architecture_var.set(ARCH)
for k in ("minimax_dit", "minimax_vae", "minimax_text_encoder"):
    app.prefs_vars[k].set("")   # clear -> should be flagged missing
_captured = {}
G.messagebox.showerror = lambda title, msg, *a, **k: _captured.setdefault("msg", msg)
ok = app.validate_inputs()
joined = _captured.get("msg", "")
ck("validate fails + flags all three empty MiniMax model paths",
   ok is False and "MiniMax H3 DiT" in joined and "Video VAE" in joined
   and "Qwen3-VL-32B" in joined, joined[:200])
# Put them back: leaving prefs_vars blank would propagate empties into anything later that
# flushes the in-memory prefs dict.
app.prefs_vars["minimax_dit"].set("X:/dit.safetensors")
app.prefs_vars["minimax_vae"].set("X:/vae.safetensors")
app.prefs_vars["minimax_text_encoder"].set("X:/te.safetensors")

# --- regressions: preset row-swap + last-train architecture (reported 4 Aug) ----------------
# 1) Setting a combobox programmatically does NOT fire <<ComboboxSelected>>, so a preset that
#    changed Network Type used to leave the old rows up: "LoRA (standard)" with the LoKR Factor
#    box still underneath it.
app.architecture_var.set("Krea 2")
app.on_architecture_changed()
app.entries["NETWORK_TYPE"].set("LoKR (Kronecker)")
app._on_network_type_changed()
_factor_row = app.rows["LOKR_FACTOR"]["entry"]
ck("LoKR selected -> Factor row shown", bool(_factor_row.winfo_manager()))
app._apply_preset_values({"NETWORK_TYPE": "LoRA (standard)", "NETWORK_DIM": 32})
ck("a preset switching to LoRA hides the Factor row (no stale row)",
   not _factor_row.winfo_manager() and bool(app.rows["NETWORK_DIM"]["entry"].winfo_manager()))
app._apply_preset_values({"NETWORK_TYPE": "LoKR (Kronecker)"})
ck("a preset switching to LoKR shows the Factor row", bool(_factor_row.winfo_manager()))

# 2) The last-train snapshot must carry the FAMILY. Presets deliberately don't (a Krea 2 preset
#    must not hijack your model choice), but "restore my last launch" plainly includes which
#    model it was — otherwise the button silently leaves you on the wrong trainer.
import json as _json  # noqa: E402
import tempfile as _tf  # noqa: E402
G.LAST_TRAIN_FILE = os.path.join(_tf.mkdtemp(prefix="lt_"), "last_train.json")
app.architecture_var.set(ARCH)
app.on_architecture_changed()
# The save is guarded by FIZGIG_NO_PERSIST, so exercising it means lifting the guard — but ONLY
# around this one call, and with LAST_TRAIN_FILE already redirected to a temp dir and save_prefs
# neutered above. Restored in finally so nothing later runs unguarded.
_real_guard = G._persist_disabled
try:
    G._persist_disabled = lambda: False
    app._save_last_train_settings()
finally:
    G._persist_disabled = _real_guard
_snap = _json.load(open(G.LAST_TRAIN_FILE, encoding="utf-8"))
ck("last-train snapshot records the architecture", _snap.get("__architecture__") == ARCH,
   _snap.get("__architecture__"))
app.architecture_var.set("Krea 2")
app.on_architecture_changed()
G.messagebox.showinfo = lambda *a, **k: None
app._load_last_train_settings()
ck("Load Settings From Last Train restores the family", app.architecture_var.get() == ARCH,
   app.architecture_var.get())
ck("an older snapshot without the key still loads (no crash)",
   (app._apply_preset_values({"NETWORK_DIM": 16}) is None))

# --- the single-frame preview caveat, under the Base Model selector ------------------------------
_note = getattr(app, "_minimax_sample_note", None)
ck("the MiniMax preview note exists", _note is not None)
if _note is not None:
    _seen = []
    for _a in (ARCH, "Krea 2", "Flux 2 Klein Base 9B", ARCH):
        app.architecture_var.set(_a)
        app.on_architecture_changed()
        _seen.append(bool(_note.winfo_manager()))
    ck("note shows for MiniMax and hides for every other family",
       _seen == [True, False, False, True], _seen)
    _txt = _note.cget("text").lower()
    # The note is deliberately SHORT (a long one goes unread), so pin the two claims it has to
    # make rather than any particular phrasing: previews are for likeness not quality, and
    # ComfyUI is where quality is judged — reachable mid-run via Pause.
    ck("it says previews track likeness, not quality",
       "likeness" in _txt and "quality" in _txt, _txt)
    ck("it points at ComfyUI, via Pause", "comfyui" in _txt and "pause" in _txt, _txt)
    ck("it stays short enough to actually be read", len(_txt) < 300, len(_txt))
    app.architecture_var.set(ARCH)
    app.on_architecture_changed()

# --- live sample override reaches the MiniMax trainer --------------------------------------------
import json as _json2  # noqa: E402
from fizgig.minimax.trainer import read_sample_override as _rso  # noqa: E402

app.architecture_var.set(ARCH)
app.on_architecture_changed()
_ovp = app._sample_override_path()
app.sample_override_var.set(True)
app.sample_override_prompt_var.set("a portrait of zwxem, smiling")
app.sample_override_seed_var.set("77")
app.sample_override_w_var.set("1024")
app.sample_override_h_var.set("768")
app._on_sample_override_changed()
ck("the GUI writes .sample_override.json", os.path.exists(_ovp))
_got = _rso(os.path.dirname(_ovp))
ck("the MiniMax trainer reads back exactly what was written",
   _got == {"prompt": "a portrait of zwxem, smiling", "seed": 77, "width": 1024, "height": 768},
   _got)
app.sample_override_var.set(False)
app._on_sample_override_changed()
ck("switching it off removes the file (samples fall back to the tab)", not os.path.exists(_ovp))
ck("a prompt-less override is inactive for MiniMax (no ref-image concept)",
   _rso(os.path.dirname(_ovp)) is None)

# the reference-image picker is a Klein edit feature — hidden for both native families
_shown = {}
for _a in ("Flux 2 Klein Base 9B", ARCH, "Krea 2"):
    app.architecture_var.set(_a)
    app.on_architecture_changed()
    _shown[_a] = bool(app._override_ref_browse_btn.winfo_manager())
ck("override ref picker: shown for Klein, hidden for MiniMax and Krea 2",
   _shown == {"Flux 2 Klein Base 9B": True, ARCH: False, "Krea 2": False}, _shown)
app.architecture_var.set(ARCH)
app.on_architecture_changed()

root.destroy()

# --- the RESTORED-family startup path ---------------------------------------------------------
# Everything above reaches MiniMax by SWITCHING family, which fires <<ComboboxSelected>> and
# applies the built-in preset. A returning user never takes that path: the architecture is
# restored from last_used["architecture"] and _arch_last_selected is seeded to the same value,
# specifically so no change event fires. If the startup preset apply were ever removed or
# reordered, the widgets would keep the construction-time settings dict instead — where
# OPTIMIZER_TYPE is adamw8bit, because that key is shared with Klein and Krea 2 and must stay
# their default. A returning MiniMax user would then silently train on the 8-bit optimizer:
# no error, no warning, just the likeness ceiling back.
_real_lu = G.load_last_used
G.load_last_used = lambda: dict(_real_lu(), architecture=ARCH_OLD)   # a pre-3.6.1 save
try:
    root2 = tk.Tk()
    root2.withdraw()
    app2 = G.LoRATrainerGUI(root2)
    ck("a pre-3.6.1 save still comes up in MiniMax, not Klein", app2._is_minimax_arch(),
       app2.architecture_var.get())
    ck("...shown under the current name", app2.architecture_var.get() == ARCH,
       app2.architecture_var.get())
    ck("a restored MiniMax session comes up in MiniMax", app2._is_minimax_arch(),
       app2.architecture_var.get())
    ck("...with adamw already selected, NOT the shared adamw8bit default",
       app2.entries["OPTIMIZER_TYPE"].get() == "adamw", app2.entries["OPTIMIZER_TYPE"].get())
    ck("...and the low-noise share applied (mid-concentrated retired)",
       app2.entries["MINIMAX_LOWNOISE_PCT"].get() == "60",
       app2.entries["MINIMAX_LOWNOISE_PCT"].get())
    ck("...and the experiment knobs off",
       app2.entries["MINIMAX_TRAIN_ADALN"].get() is False
       and app2.minimax_distill_var.get() is False,
       (app2.entries["MINIMAX_TRAIN_ADALN"].get(), app2.minimax_distill_var.get()))
    root2.destroy()
finally:
    G.load_last_used = _real_lu

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
