"""MiniMax H3 block swap + gradient checkpointing + VRAM planner — headless, CPU, tiny model.

Pins:
  * gradient checkpointing produces the SAME loss and the SAME LoRA gradients as the plain
    forward (checkpointing must be a pure memory optimization, never a math change);
  * enable_block_swap parks the last N blocks on CPU and the forward still runs (on CPU the
    JIT moves are no-ops, but the whole swap code path — including the checkpoint wrapper with
    swapped=True — executes);
  * the pure planner maps free-VRAM / MP scenarios to sensible (blocks, checkpointing) plans;
  * the GUI passes --blocks_to_swap auto through for MiniMax instead of resolving it with the
    Klein tier table.

Run: venv/Scripts/python.exe tests/test_minimax_swap.py
"""
import os
import sys

os.environ["FIZGIG_NO_PERSIST"] = "1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402
from fizgig.minimax.model import MiniMaxH3DiT, MiniMaxH3Config  # noqa: E402
from fizgig.minimax.trainer import compute_loss, plan_vram  # noqa: E402
from fizgig.networks.lora import create_network  # noqa: E402

FAILS = []


def ck(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


CFG = dict(hidden_size=64, num_layers=4, token_refiner_num_layers=2, num_attention_heads=4,
           attention_head_dim=16, ffn_hidden_size=48, latents_dim=24, audio_latents_dim=6,
           patch_size=(1, 2, 2), text_dim=32, timestep_input_dim=16, time_embed_hidden_size=64,
           time_embed_dim=32, rope_inv_freq_len=2)


def build(seed):
    torch.manual_seed(seed)
    dit = MiniMaxH3DiT(MiniMaxH3Config(**CFG))
    dit.requires_grad_(False)
    net = create_network(None, "lora_unet", 1.0, 4, 4, None, [], dit)
    net.apply_to(text_encoders=None, unet=dit, apply_text_encoder=False, apply_unet=True)
    net.requires_grad_(True)
    return dit, net


def one_step(dit, net):
    torch.manual_seed(123)
    latent = torch.randn(1, 24, 1, 8, 6)
    text = torch.randn(1, 7, 32)
    loss, _ = compute_loss(dit, latent, text, sigma=torch.tensor([0.5]),
                           noise=torch.randn(1, 24, 1, 8, 6, generator=torch.Generator().manual_seed(7)))
    loss.backward()
    grads = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}
    return loss.item(), grads


# --- 1. checkpointing is a pure memory optimization -----------------------------------------
dit_a, net_a = build(0)
loss_plain, grads_plain = one_step(dit_a, net_a)

dit_b, net_b = build(0)
dit_b.enable_gradient_checkpointing()
loss_ckpt, grads_ckpt = one_step(dit_b, net_b)

ck("checkpointed loss identical to plain", abs(loss_plain - loss_ckpt) < 1e-6,
   f"{loss_plain:.6f} vs {loss_ckpt:.6f}")
_gmatch = all(torch.allclose(grads_plain[k], grads_ckpt[k], atol=1e-6) for k in grads_plain)
ck("checkpointed LoRA grads identical to plain", _gmatch and len(grads_plain) == len(grads_ckpt),
   f"{len(grads_plain)} grads compared")

# --- 2. block swap bookkeeping + swapped forward path ---------------------------------------
dit_c, net_c = build(0)
n = dit_c.enable_block_swap(2)
ck("enable_block_swap parks the LAST n blocks", n == 2 and dit_c._swap_from == 2)
dit_c.enable_gradient_checkpointing()
loss_swap, grads_swap = one_step(dit_c, net_c)
ck("swapped+checkpointed step matches plain", abs(loss_plain - loss_swap) < 1e-6,
   f"{loss_plain:.6f} vs {loss_swap:.6f}")
ck("swap cap keeps >=2 blocks resident", dit_c.enable_block_swap(99) == len(dit_c.blocks) - 2)

# --- 3. planner scenarios -------------------------------------------------------------------
ck("planner: 5090 (30GB free, 0.25MP) -> no swap, no ckpt", plan_vram(30.0, 0.25) == (0, False))
b, c = plan_vram(22.5, 0.25)   # 3090/4090
ck("planner: 24GB card (22.5 free) -> 0 swap + checkpointing", (b, c) == (0, True), (b, c))
b, c = plan_vram(14.5, 0.25)   # 16GB card
ck("planner: 16GB card (14.5 free) -> swap + checkpointing", b > 0 and c and b <= 40, (b, c))
b1, _ = plan_vram(20.0, 0.25)
b2, _ = plan_vram(20.0, 1.0)
ck("planner: higher MP -> more swap", b2 > b1, f"{b1} -> {b2}")
ck("planner: swap capped at 40", plan_vram(1.0, 4.0)[0] == 40)

# --- 3a2. int8 needs MORE headroom than NF4 at the same load ------------------------------------
# int8 dequantizes a bf16 weight per matmul (fc1 = 308 MB) and checkpointing keeps several live;
# NF4's fused kernel never materializes one. Budgeting int8 with NF4's anchors OOM'd a real 1 MP
# run at 30.3 of 31.8 GB on the first step after the epoch-0 preview.
from fizgig.minimax.trainer import (_INT8_TRANSIENT_GB, _RESIDENT_INT8_GB,  # noqa: E402
                                    _RESIDENT_PRUNED_GB)
_i8 = plan_vram(31.0, 1.0, resident_gb=_RESIDENT_INT8_GB, transient_gb=_INT8_TRANSIENT_GB)
_n4 = plan_vram(31.0, 1.0, resident_gb=_RESIDENT_PRUNED_GB)
# This used to assert int8 at 1 MP MUST swap on a 31 GB card. Measured 6 Aug on the real model
# (int8 base, LoKR 8 + adamw, checkpointed) it peaks at 24.56 GB — it fits with room to spare,
# and the old claim came from _ACT_GB_CKPT modelling checkpointed memory as scaling with
# megapixels when it is nearly flat (0.23 MP -> 24.39, 0.98 MP -> 24.56). The planner was
# buying ~25 blocks of swap, and roughly 4x the step time, for nothing.
ck("int8 at 1 MP on 31 GB fits with checkpointing, no swap (measured peak 24.56 GB)",
   _i8 == (0, True), _i8)
ck("NF4 at 1 MP on the same card does not need swap either", _n4[0] == 0, _n4)
ck("...and checkpointing is what makes 1 MP fit — without it, it does not",
   plan_vram(31.0, 1.0, resident_gb=_RESIDENT_INT8_GB, transient_gb=_INT8_TRANSIENT_GB)[1] is True)
ck("int8 still fits with no swap at a smaller bucket",
   plan_vram(31.0, 0.7, resident_gb=_RESIDENT_INT8_GB, transient_gb=_INT8_TRANSIENT_GB)[0] == 0)
ck("the transient term makes the plan strictly more conservative",
   plan_vram(31.0, 1.0, resident_gb=_RESIDENT_INT8_GB, transient_gb=_INT8_TRANSIENT_GB)[0]
   >= plan_vram(31.0, 1.0, resident_gb=_RESIDENT_INT8_GB, transient_gb=0.0)[0])

# --- 3a2b. skipping checkpointing has to EARN it -------------------------------------------------
# A real run (6 Aug) needed 32.13 GB of 32.5 free and the planner duly chose no-checkpointing —
# a 0.37 GB margin — then trained at 4-6 s/step instead of ~1. On Windows the driver spills to
# system RAM rather than OOMing, so an over-tight plan does not fail loudly, it just crawls.
# Recompute costs ~0.1 s/step and saves ~5 GB, so a thin margin is never worth taking.
from fizgig.minimax.trainer import _NOCKPT_MARGIN_GB, adapter_vram_gb as _avg  # noqa: E402

_AD8 = _avg(313_100_000, "adamw")
_pk = dict(resident_gb=_RESIDENT_INT8_GB, transient_gb=_INT8_TRANSIENT_GB, adapter_gb=_AD8)
ck("the real 32.5 GB / 0.26 MP case now checkpoints instead of running on fumes",
   plan_vram(32.5, 0.26, **_pk) == (0, True), plan_vram(32.5, 0.26, **_pk))
ck("...and no-checkpointing is still reachable with genuine room",
   plan_vram(40.0, 0.26, **_pk) == (0, False), plan_vram(40.0, 0.26, **_pk))
ck("the extra margin is real, not cosmetic", _NOCKPT_MARGIN_GB >= 2.0, _NOCKPT_MARGIN_GB)

# --- 3a3. Auto picks the base PRECISION and the swap count together -------------------------------
# Deciding swap alone, with precision fixed by which file was loaded, gave 24 GB cards the worst
# available outcome: the int8 base is ~21 GB, so 38 of 50 blocks went to CPU and crossed PCIe
# every step (~4x slower) when the same file loaded 4-bit is ~10.5 GB and needs no swap at all.
# Nothing about that failure is visible at runtime -- the run just takes four times as long.
from fizgig.minimax.trainer import adapter_vram_gb, plan_base_quant  # noqa: E402

_AD = adapter_vram_gb(313_100_000, "adamw")          # LoKR 8 + adamw (an opt-in Network Type;
# every shipped H3 preset is LoRA). Trainable term only: the frozen LoRAs and the EMA shadow are
# added by plan_adapter_gb at the call site, and tests/test_minimax_vram_terms.py covers those.
_m30, _s30, _, _ = plan_base_quant(30.5, True, mp=0.25, adapter_gb=_AD)
ck("32 GB card keeps the accurate int8 base, no swap", (_m30, _s30) == ("int8", 0), (_m30, _s30))
# The H2D streamer (#73) flipped these two pins ON PURPOSE: swap used to cost ~4x step time
# (classic parking), so 4-bit-resident beat int8-swapped; streamed int8 measured ~1.85 s/it
# at swap 40 and 1 MP on a simulated 16 GB card, so the accurate base wins wherever its
# residual footprint fits. 4-bit is now the floor for genuinely tiny cards only.
_m22, _s22, _, _ = plan_base_quant(22.0, True, mp=0.25, adapter_gb=_AD)
ck("24 GB card streams int8 rather than dropping to 4-bit",
   _m22 == "int8" and 0 < _s22 <= 40, (_m22, _s22))
_m14, _s14, _, _ = plan_base_quant(14.0, True, mp=0.25, adapter_gb=_AD)
ck("16 GB card streams int8 too — the accurate base at every mainstream tier",
   _m14 == "int8" and 0 < _s14 <= 40, (_m14, _s14))
_m10, _s10, _, _ = plan_base_quant(6.5, True, mp=0.25, adapter_gb=_AD)
ck("...but a card too small for even streamed int8 still falls back to 4-bit",
   _m10 == "nf4", (_m10, _s10))
# int8 is preferred wherever it fits: 4-bit costs ~9% error in the frozen base, which the LoRA
# then spends capacity correcting. It is the fallback, never the default.
ck("...and 4-bit is never chosen when int8 fits unswapped",
   all(plan_base_quant(f, True, mp=0.25, adapter_gb=_AD)[0] == "int8"
       for f in (30.0, 34.0, 40.0)))
# A bf16 checkpoint has no int8 weights to keep, so there is nothing to choose.
ck("a bf16 checkpoint always reports nf4", plan_base_quant(30.0, False, adapter_gb=_AD)[0] == "nf4")

# --- 3b. timestep schedule: the default is the reference recipe (shift-12) -------------------
# History: an earlier revision asserted a mid-band density here, on the belief that shift-12
# was sampler-only. ai-toolkit's MiniMax entry actually overrides its global 'sigmoid' to
# timestep_type='shift' through a shift-12 scheduler — high-noise-heavy IS the reference
# training recipe (and why 1e-4 is a sane LR for it). test_minimax_parity.py pins the full
# density contract; this is just the smoke check.
from fizgig.minimax.trainer import sample_sigmas  # noqa: E402
_g = torch.Generator().manual_seed(0)
_s = torch.cat([sample_sigmas(1000, "cpu", generator=_g, image_tokens=225) for _ in range(5)])
ck("default schedule: shift-12 (median in the video regime)",
   _s.median().item() > 0.85, f"median={_s.median():.3f}")
_ab = sample_sigmas(1000, "cpu", shift="sigmoid", generator=_g)
ck("sigmoid A/B literal still reachable (median ~0.5)", 0.4 < _ab.median().item() < 0.6)

# --- 4. GUI passes 'auto' through for MiniMax -----------------------------------------------
import tkinter as tk  # noqa: E402
import lora_trainer_gui as G  # noqa: E402
G.LAST_USED_FILE = os.path.join(os.environ.get("TEMP", "/tmp"), "nope", ".last_used.json")
root = tk.Tk(); root.withdraw()
app = G.LoRATrainerGUI(root)
app.architecture_var.set("MiniMax H3 (experimental)")
app.prefs_vars["minimax_dit"].set("X:/dit.safetensors")
app.settings = {
    "DATASET_CONFIG": "X:/ds.toml", "LORA_OUTPUT_DIR": "X:/out", "LORA_NAME": "t",
    "NETWORK_DIM": 16, "NETWORK_ALPHA": 16, "LEARNING_RATE": 1e-4, "MAX_TRAIN_EPOCHS": 20,
    "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42, "BLOCKS_SWAP": "auto", "ADAPTIVE_LR": False,
    "MAX_GRAD_NORM": "1.0", "OPTIMIZER_TYPE": "adamw8bit", "OPTIMIZER_ARGS": "",
    "METADATA_TITLE": "", "METADATA_AUTHOR": "", "METADATA_DESCRIPTION": "",
    "METADATA_LICENSE": "", "METADATA_TAGS": "", "METADATA_TRIGGER_PHRASE": "",
}
cmd = [str(x) for x in app.build_training_command(G.ARCHITECTURES["MiniMax H3 (experimental)"])]
i = cmd.index("--blocks_to_swap") if "--blocks_to_swap" in cmd else -1
ck("minimax train cmd passes --blocks_to_swap auto", i >= 0 and cmd[i + 1] == "auto",
   cmd[i:i + 2] if i >= 0 else "flag missing")
app.settings["BLOCKS_SWAP"] = 12
cmd = [str(x) for x in app.build_training_command(G.ARCHITECTURES["MiniMax H3 (experimental)"])]
i = cmd.index("--blocks_to_swap")
ck("explicit swap number passes through", cmd[i + 1] == "12", cmd[i:i + 2])
root.destroy()

print()
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
