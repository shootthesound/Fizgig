"""Per-family LoRA name suffix + shared sample resolutions — headless, no GPU.

Run: venv/Scripts/python.exe tests/test_lora_name_suffix.py
"""
import os
import sys
import tempfile

os.environ["FIZGIG_NO_PERSIST"] = "1"
REPO = r"W:/Peter/Documents/Development/Fizgig"
sys.path.insert(0, REPO)

import tkinter as tk
import lora_trainer_gui as G

G.LAST_USED_FILE = os.path.join(os.environ["TEMP"], "nope", ".last_used.json")

KLEIN = "Flux 2 Klein Base 9B"
KREA = "Krea 2"
fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


root = tk.Tk()
root.withdraw()
g = G.LoRATrainerGUI(root)
# A clean output folder: the retag refuses to touch a paused run's name (a
# .fizgig_paused.json in the output folder blocks it, by design), and a stale one in
# the machine's real output_loras made all five retag pins fail (found 4 Sep).
g.settings["LORA_OUTPUT_DIR"] = tempfile.mkdtemp(prefix="fizgig_suffix_")


def set_name(n):
    e = g.entries["LORA_NAME"]
    e.delete(0, tk.END)
    e.insert(0, n)


def name():
    return g.entries["LORA_NAME"].get()


def switch(arch):
    g.architecture_var.set(arch)
    g._on_architecture_selected()


# --- 1. only an EXACT trailing family suffix is ever touched -----------------------------
_retag = [("myface_k9b", KREA, "myface_krea2"),
          ("myface_krea2", KLEIN, "myface_k9b"),
          ("LoraName_TokenName_k9b", KREA, "LoraName_TokenName_krea2"),
          ("a_b_c_k9b", KREA, "a_b_c_krea2")]
for _n, _t, _w in _retag:
    set_name(_n)
    g._apply_lora_name_suffix(_t)
    ck(f"retag {_n!r}", name() == _w, name())

# Anything that is not exactly "<something>_<known suffix>" must survive verbatim. These are
# the cases that would break a user's own naming scheme if the match were sloppy.
for _n in ("bobs_dog", "portrait_v2", "myface", "k9b", "krea2", "myfacek9b",
           "myface_K9B", "my_k9b_face", "face_k9b2", "_k9b_", "v2_k9b_final"):
    set_name(_n)
    g._apply_lora_name_suffix(KREA)
    ck(f"  untouched: {_n!r}", name() == _n, name())

# Idempotent — safe to call at startup and on every switch.
set_name("myface_krea2")
g._apply_lora_name_suffix(KREA)
ck("already-correct suffix is a no-op", name() == "myface_krea2", name())

# --- 2. end-to-end through a real family switch -------------------------------------------
switch(KLEIN)
set_name("myface_k9b")
switch(KREA)
ck("switch Klein->Krea retags the name", name() == "myface_krea2", name())
switch(KLEIN)
ck("switch back retags it home", name() == "myface_k9b", name())

# A non-conventional name must survive a switch. The memory has to be empty for the DESTINATION
# family, or it will (correctly) restore that family's own remembered name and mask this.
switch(KREA)
set_name("bobs_dog")
g._arch_settings_memory.clear()          # make Klein a "first visit" again
switch(KLEIN)
ck("a non-conventional name survives a real switch", name() == "bobs_dog", name())
g._arch_settings_memory.clear()

# --- 3. per-family memory still wins ------------------------------------------------------
switch(KLEIN)
set_name("klein_subject_k9b")
switch(KREA)
set_name("krea_subject_krea2")
switch(KLEIN)
ck("per-family memory restores the Klein name", name() == "klein_subject_k9b", name())
switch(KREA)
ck("per-family memory restores the Krea 2 name", name() == "krea_subject_krea2", name())

# --- 4. never rename out from under a live or paused run ----------------------------------
class _LiveProc:
    def poll(self):
        return None          # still running


switch(KLEIN)
set_name("myface_k9b")
g.current_process = _LiveProc()
ck("blocked while a run is live", g._lora_name_rename_blocked() is True)
g._apply_lora_name_suffix(KREA)
ck("  name untouched while a run is live", name() == "myface_k9b", name())
g.current_process = None

# _paused_sidecar_path reads settings["LORA_OUTPUT_DIR"] — the dir of the run that was actually
# LAUNCHED (snapshotted at start_training, and seeded from last_used at startup), not whatever is
# currently typed in the box. That is the right source: the pause belongs to that launch.
_paused_dir = tempfile.mkdtemp()
g.settings["LORA_OUTPUT_DIR"] = _paused_dir
open(os.path.join(_paused_dir, ".fizgig_paused.json"), "w").write("{}")
ck("blocked while a paused run exists", g._lora_name_rename_blocked() is True)
g._apply_lora_name_suffix(KREA)
ck("  name untouched while paused — a rename would orphan the state dir",
   name() == "myface_k9b", name())
os.remove(os.path.join(_paused_dir, ".fizgig_paused.json"))
ck("unblocked once the pause is cleared", g._lora_name_rename_blocked() is False)

# --- 5. shared resolution list ------------------------------------------------------------
ck("Samples tab uses the shared list",
   list(g.sample_width_combo.cget("values")) == G.SAMPLE_RESOLUTIONS
   and list(g.sample_height_combo.cget("values")) == G.SAMPLE_RESOLUTIONS)
ck("  the list reaches 1536", "1536" in G.SAMPLE_RESOLUTIONS and "1280" in G.SAMPLE_RESOLUTIONS)


def _override_res_widgets():
    """The override panel's comboboxes aren't stored on self — find them by their var."""
    found = []

    def walk(w):
        for c in w.winfo_children():
            try:
                if str(c.cget("textvariable")) in (str(g.sample_override_w_var),
                                                   str(g.sample_override_h_var)):
                    found.append(c)
            except Exception:
                pass
            walk(c)

    walk(g.master)
    return found


_ov = _override_res_widgets()
ck("found both override resolution widgets", len(_ov) == 2, len(_ov))
ck("  override offers exactly what the Samples tab does",
   all(list(w.cget("values")) == G.SAMPLE_RESOLUTIONS for w in _ov),
   [list(w.cget("values")) for w in _ov])

root.destroy()
print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S)"))
sys.exit(1 if fails else 0)
