"""Weight averaging (EMA) on Krea 2 (9 Sep 2026): the shared EMAWeights class, the Krea 2 GUI row
(Training Parameters, Krea 2 only), the launch dict, the command builder, presets, trainer
signature and CLI. Headless (FIZGIG_NO_PERSIST); no GPU.

Run: venv/Scripts/python.exe tests/test_krea2_ema.py
"""
import hashlib
import io as _io
import os
import sys

os.environ["FIZGIG_NO_PERSIST"] = "1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


# --- the shared class: swap round-trip is exact, the average moves, state round-trips ----------
from fizgig.training.ema import EMAWeights  # noqa: E402
from fizgig.minimax.trainer import EMAWeights as _H3EMA  # noqa: E402
ck("H3 trainer re-exports the shared class (no behaviour change there)", _H3EMA is EMAWeights)
net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
ema = EMAWeights(net, 0.98)
raw0 = [p.detach().clone() for p in net.parameters()]
for _ in range(5):
    for p in net.parameters():
        p.data.add_(0.1)
    ema.update()
avg = [s.clone() for s in ema.shadow]
ck("the average lags the raw weights (it is an average, not a copy)",
   all(not torch.allclose(a, p.detach().float()) for a, p in zip(avg, net.parameters())))
ck("...and has moved from the start", all(not torch.allclose(a, r.float()) for a, r in zip(avg, raw0)))
raw_now = [p.detach().clone() for p in net.parameters()]
ema.swap_in()
ck("swap_in puts the average into the network",
   all(torch.allclose(p.detach().float(), a) for p, a in zip(net.parameters(), avg)))
ema.swap_out()
ck("swap_out restores the raw weights exactly",
   all(torch.equal(p.detach(), r) for p, r in zip(net.parameters(), raw_now)))
sd = ema.state_dict(); ema2 = EMAWeights(net, 0.98); ema2.load_state_dict(sd)
ck("state dict round-trips (n and shadow)", ema2.n == ema.n and all(torch.equal(a, b) for a, b in zip(ema2.shadow, ema.shadow)))

# --- trainer + CLI plumbing -----------------------------------------------------------------
import inspect  # noqa: E402
from fizgig.krea2 import trainer as KT  # noqa: E402
ck("train_krea2 takes ema_decay (default 0 = off)",
   "ema_decay" in inspect.signature(KT.train_krea2).parameters
   and inspect.signature(KT.train_krea2).parameters["ema_decay"].default == 0.0)
ck("_write_state_files takes ema (ema.pt beside the raw lora.safetensors)",
   "ema" in inspect.signature(KT._write_state_files).parameters)
src_t = _io.open(os.path.join(REPO, "src", "fizgig", "krea2", "trainer.py"), encoding="utf-8").read()
ck("both optimizer steps update the average", src_t.count("ema.update()") == 2)
ck("checkpoint, both preview temps and the final LoRA are saved from the average",
   src_t.count("ema.swap_in()") == 4 and src_t.count("ema.swap_out()") == 3)
ck("resume restores ema.pt", 'os.path.join(resume_state_dir, "ema.pt")' in src_t)
ck("the state saver passes the average on BOTH its write calls (normal and retry)",
   src_t.count("dtype=dtype, extra=extra, ema=ema)") == 2)
# and the writer really puts ema.pt beside the raw weights
import tempfile
_net = torch.nn.Sequential(torch.nn.Linear(4, 4))
from safetensors.torch import save_file as _sf
_net.save_weights = lambda path, dtype, metadata: _sf({k: v.detach().to(dtype) for k, v in _net.state_dict().items()}, path, metadata)  # what a LoRANetwork provides
_d = tempfile.mkdtemp(prefix="k2ema_")
KT._write_state_files(_d, _net, None, epoch=1, global_step=1, network_dim=4, network_alpha=4,
                      dtype=torch.float32, extra=None, ema=EMAWeights(_net, 0.98))
ck("_write_state_files writes ema.pt beside lora.safetensors",
   os.path.exists(os.path.join(_d, "ema.pt")) and os.path.exists(os.path.join(_d, "lora.safetensors")))
ck("final metadata records ss_ema_decay", '"ss_ema_decay"' in src_t)
cli = _io.open(os.path.join(REPO, "src", "fizgig", "scripts", "krea2_train.py"), encoding="utf-8").read()
ck("CLI: --ema_decay exists and is passed through", cli.count("--ema_decay") == 1 and "ema_decay=args.ema_decay" in cli)

# --- GUI: Krea 2-only row in Training Parameters, launch dict, builder, presets -----------------
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
root = tk.Tk(); root.geometry("1400x900+3000+3000")
app = G.LoRATrainerGUI(root); root.update()
KREA2 = next(k for k, v in G.ARCHITECTURES.items() if v.get("is_krea2"))
H3 = next(k for k, v in G.ARCHITECTURES.items() if v.get("is_minimax"))
KLEIN = next(k for k, v in G.ARCHITECTURES.items() if not v.get("is_krea2") and not v.get("is_minimax"))


def mapped(w):
    try:
        return bool(w.winfo_ismapped())
    except Exception:
        return False


def settle(arch):
    app.notebook.select(app.training_tab); app.architecture_var.set(arch); app.update_ui_for_architecture(); root.update()


try:
    settle(KREA2)
    ck("Krea 2: EMA row mapped in Training Parameters", mapped(app._krea2_ema_label) and mapped(app.entries["KREA2_EMA"]))
    ck("Krea 2: the row is in the Training Parameters section, not Other Options",
       app._krea2_ema_label.master is app.collapsible_sections["training"].get_content_frame())
    ck("Krea 2: options Off / 0.98 / 0.99 / 0.995, default Off",
       tuple(app.entries["KREA2_EMA"].cget("values")) == ("Off", "0.98", "0.99", "0.995") and app.entries["KREA2_EMA"].get() == "Off")
    settle(H3)
    ck("MiniMax: the Krea 2 row is hidden", not mapped(app._krea2_ema_label) and not mapped(app._krea2_ema_hint))
    ck("MiniMax: its own EMA control is untouched (0.98 recommended, in Other Options)",
       app.entries["MINIMAX_EMA"].get().startswith("0.98") and app.entries["MINIMAX_EMA"].master.master is app.collapsible_sections["scheduler"].get_content_frame())
    settle(KLEIN)
    ck("Klein: hidden", not mapped(app._krea2_ema_label))
    settle(KREA2)
    gui_src = _io.open(os.path.join(REPO, "lora_trainer_gui.py"), encoding="utf-8").read()
    ck("launch dict carries KREA2_EMA", '"KREA2_EMA": self.entries["KREA2_EMA"].get()' in gui_src)
    for name, preset in G.KREA2_BUILT_IN_PRESETS.items():
        ck(f"preset {name[:28]!r}... ships KREA2_EMA Off (A/B first)", preset.get("KREA2_EMA") == "Off")
    ck("MiniMax presets do not carry KREA2_EMA", all("KREA2_EMA" not in p for p in G.MINIMAX_BUILT_IN_PRESETS.values()))
    base = dict(app.settings)
    base.update({"DATASET_CONFIG": "X:/ds.toml", "LORA_OUTPUT_DIR": "X:/out", "LORA_NAME": "t", "NETWORK_DIM": 16,
                 "NETWORK_ALPHA": 16, "LEARNING_RATE": 1e-4, "MAX_TRAIN_EPOCHS": 3, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 1,
                 "MAX_GRAD_NORM": "1.0", "OPTIMIZER_TYPE": "adamw8bit", "OPTIMIZER_ARGS": ""})
    app.settings = dict(base, KREA2_EMA="0.98")
    c = [str(x) for x in app._build_krea2_train_command()]
    ck("builder emits --ema_decay 0.98", "--ema_decay" in c and c[c.index("--ema_decay") + 1] == "0.98")
    app.settings = dict(base, KREA2_EMA="Off")
    c = [str(x) for x in app._build_krea2_train_command()]
    ck("...and nothing when Off", "--ema_decay" not in c)
finally:
    try:
        root.destroy()
    except Exception:
        pass
ck("prefs.json / last_used.json byte-identical after the run", before == _digests())

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}"); sys.exit(1)
print("ALL PASS")
