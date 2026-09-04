"""Phase B of H3-in-the-workbench: block layout, engine wiring, bake mapping, GUI tri-state.

Silent failures pinned: a block regex that lets h3blk_2 touch blocks_20 (a slider that edits
the wrong block LOOKS fine), the zero-init trap (a LoRA that applies but contributes nothing),
the LoKR whitelist gap (LyCORIS modules skipped on quantized bases — loads "successfully" with
0 modules), the bake mapper calling H3 keys block_N (sliders silently ignored at save), and a
prompt disk-cache that misses and quietly reloads the 32B TE every session.

Run: venv/Scripts/python.exe tests/test_h3_engine.py
"""

import hashlib
import os
import re
import sys
import tempfile

os.environ["FIZGIG_NO_PERSIST"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

_fails = []


def ck(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (f"  {detail}" if detail else ""))
    if not ok:
        _fails.append(name)


TMP = tempfile.mkdtemp(prefix="fizgig_h3_eng_")

# --- block layout ---------------------------------------------------------------------------------
from fizgig.repair_studio.h3_blocks import (  # noqa: E402
    all_block_ids_h3, block_regex_h3, extract_block_ids_h3)
from fizgig.repair_studio.state import SliderState  # noqa: E402

ids = all_block_ids_h3()
ck("52 block ids: 50 main + 2 refiner", len(ids) == 52 and ids[0] == "h3blk_0"
   and ids[49] == "h3blk_49" and ids[50] == "h3_rf_0")

_r2 = block_regex_h3("h3blk_2")
ck("h3blk_2 matches blocks_2 only",
   re.search(_r2, "lora_unet_blocks_2_attn_qkv_proj") is not None
   and re.search(_r2, "lora_unet_blocks_20_attn_qkv_proj") is None
   and re.search(_r2, "lora_unet_blocks_21_mlp_fc1") is None
   and re.search(_r2, "lora_unet_token_refiner_blocks_2_attn_qkv_proj") is None)
_rr = block_regex_h3("h3_rf_1")
ck("h3_rf_1 matches only the refiner",
   re.search(_rr, "lora_unet_token_refiner_blocks_1_mlp_fc1") is not None
   and re.search(_rr, "lora_unet_blocks_1_mlp_fc1") is None)

st = SliderState.default_h3()
ck("default_h3: 52 blocks at 768x768",
   len(st.blocks) == 52 and st.preview_width == 768 and st.preview_height == 768)

# --- the bake mapper: H3 keys resolve against the STATE's namespace -------------------------------
from fizgig.repair_studio.bake import _block_id_from_key  # noqa: E402

h3_ids = set(st.blocks.keys())
ck("bake maps H3 main keys to h3blk_N when the state is H3",
   _block_id_from_key("lora_unet_blocks_3_mlp_fc1.lora_up.weight".split(".")[0], h3_ids) == "h3blk_3")
ck("...and to block_N for a Krea 2 state (shared raw key shape)",
   _block_id_from_key("lora_unet_blocks_3_mlp_fc1", {"block_3"}) == "block_3")
ck("...refiner keys map to h3_rf_N",
   _block_id_from_key("lora_unet_token_refiner_blocks_1_attn_qkv_proj", h3_ids) == "h3_rf_1")
ck("...Klein keys are untouched",
   _block_id_from_key("lora_unet_double_blocks_4_img_attn_proj", h3_ids) == "double_4")

# --- LoRA apply on a tiny H3-shaped DiT: zero-init pin + per-block isolation ----------------------
from fizgig.minimax.model import MiniMaxH3Config, MiniMaxH3DiT  # noqa: E402
from fizgig.repair_studio.h3_engine import _apply_lora  # noqa: E402

torch.manual_seed(0)
cfg = MiniMaxH3Config(hidden_size=64, num_layers=3, token_refiner_num_layers=2,
                      num_attention_heads=4, attention_head_dim=16, ffn_hidden_size=48,
                      latents_dim=24, audio_latents_dim=6, patch_size=(1, 2, 2), text_dim=20,
                      timestep_input_dim=16, time_embed_hidden_size=64, time_embed_dim=32,
                      rope_inv_freq_len=2)
dit = MiniMaxH3DiT(cfg).eval()

# A tiny LoRA over blocks 0 and 2 + refiner 1, kohya keys matching the tiny DiT's Linears.
sd = {}
for dotted in ("blocks.0.attn.qkv_proj", "blocks.2.mlp.fc1", "token_refiner.blocks.1.mlp.fc1"):
    mod = dict(dit.named_modules())[dotted]
    name = "lora_unet_" + dotted.replace(".", "_")
    r = 4
    sd[f"{name}.lora_down.weight"] = torch.randn(r, mod.in_features) * 0.3
    sd[f"{name}.lora_up.weight"] = torch.randn(mod.out_features, r) * 0.3
    sd[f"{name}.alpha"] = torch.tensor(float(r))

net = _apply_lora(dit, sd, 1.0, "cpu", torch.float32)
ck("3 modules wired", len(net.unet_loras) == 3)
ck("weights actually loaded (the zero-init trap)",
   all(float(m.lora_up.weight.abs().sum()) > 0 for m in net.unet_loras))
ck("extract_block_ids_h3 sees them",
   extract_block_ids_h3(net) == {"h3blk_0", "h3blk_2", "h3_rf_1"})

net.set_module_multiplier_by_pattern(block_regex_h3("h3blk_2"), 0.25)
mults = {m.lora_name: m.multiplier for m in net.unet_loras}
ck("per-block multiplier isolation",
   mults["lora_unet_blocks_2_mlp_fc1"] == 0.25
   and mults["lora_unet_blocks_0_attn_qkv_proj"] == 1.0
   and mults["lora_unet_token_refiner_blocks_1_mlp_fc1"] == 1.0)

# --- the no-LoRA render (the player's third pane): every module off for that render only ----------
import threading  # noqa: E402
from fizgig.repair_studio.h3_engine import H3RepairEngine as _H3E  # noqa: E402
from fizgig.repair_studio.state import SliderState as _SS  # noqa: E402

_txt = torch.randn(1, 5, cfg.text_dim)


class _MiniEngine:
    """Just enough of H3RepairEngine for render_latent on the tiny DiT — the real method,
    bound onto a stub, so the no-LoRA path is the shipped code."""
    render_latent = _H3E.render_latent
    apply_state = _H3E.apply_state
    _reinstall_adaln = _H3E._reinstall_adaln
    _adaln_pairs_now = _H3E._adaln_pairs_now
    _block_factor = _H3E._block_factor

    def __init__(self):
        self.dit, self.primary_network, self.donor_network = dit, net, None
        self.device, self.dtype, self._steps = "cpu", torch.float32, 2
        self.resume_enabled, self._cache_device = False, "cpu"
        self._act_cache = self._act_cache_key = self._act_cache_state = None
        self.primary_path, self.donor_path, self._turbo_strength = "p", None, 1.0
        self._cancel_event, self.on_step, self.last_resume_from = threading.Event(), None, None

    def _encode_prompt(self, prompt):
        return _txt

    def set_turbo_strength(self, s):
        pass


_eng = _MiniEngine()
_st = _SS.default_h3()
_st.seed, _st.prompt, _st.preview_width, _st.preview_height, _st.preview_frames = 3, "x", 64, 64, 5
_st.blocks["h3blk_2"].primary_enabled = False          # one block off in the state itself
_lora, _ = _eng.render_latent(_st, frames=5, steps=2)
_nolora, _ = _eng.render_latent(_st, frames=5, steps=2, no_lora=True)
_flags = {m.lora_name: m.enabled for m in net.unet_loras}
_off = _SS.default_h3()
_off.seed, _off.prompt, _off.preview_width, _off.preview_height, _off.preview_frames = 3, "x", 64, 64, 5
for _bs in _off.blocks.values():
    _bs.primary_enabled = False
_alloff, _ = _eng.render_latent(_off, frames=5, steps=2)
ck("no_lora render == every block disabled by hand (the base model alone)",
   torch.equal(_nolora, _alloff))
ck("...and differs from the LoRA'd render", not torch.equal(_lora, _nolora))
ck("after a no_lora render the state's flags are back (block 2 off, block 0 + refiner on)",
   _flags["lora_unet_blocks_2_mlp_fc1"] is False and _flags["lora_unet_blocks_0_attn_qkv_proj"] is True
   and _flags["lora_unet_token_refiner_blocks_1_mlp_fc1"] is True, _flags)
_again, _ = _eng.render_latent(_st, frames=5, steps=2)
ck("...and the next LoRA'd render is bit-identical to the one before it", torch.equal(_lora, _again))

# --- the pass-1 cache is sized per render and parked on the CPU when the GPU can't hold it -------
import fizgig.utils.device as _devmod  # noqa: E402


class _BigDiT:
    class config:
        hidden_size = 5376
    blocks = [None] * 50


_big = _MiniEngine()
_big.dit, _big._cache_device = _BigDiT(), "cuda"
_big._RESUME_HEADROOM_GB = _H3E._RESUME_HEADROOM_GB
_big._RESUME_HEADROOM_FRAC = _H3E._RESUME_HEADROOM_FRAC
_big._RESUME_GPU_MAX_GB = _H3E._RESUME_GPU_MAX_GB
_big.resume_cache_gb = _H3E.resume_cache_gb.__get__(_big)
_big._resume_cache_device = _H3E._resume_cache_device.__get__(_big)
_gb56 = _big.resume_cache_gb(768, 768, 56)
_gb22 = _big.resume_cache_gb(768, 640, 22)
_gbd = _big.resume_cache_gb(512, 416, 22)
ck("pass-1 cache estimate: 768x768x56 ≈ 5.1 GB, 768x640x22 ≈ 1.9 GB, dial 512x416x22 ≈ 0.9 GB",
   4.8 < _gb56 < 5.4 and 1.7 < _gb22 < 2.1 and 0.7 < _gbd < 1.1, (round(_gb56, 2), round(_gb22, 2), round(_gbd, 2)))
_free_orig = _devmod.plannable_free_vram
try:
    _devmod.plannable_free_vram = lambda: 8.0
    ck("8 GB free: the 22-frame caches record on the GPU, the 56-frame one is parked on the CPU",
       _big._resume_cache_device(768, 640, 22) == "cuda" and _big._resume_cache_device(512, 416, 22) == "cuda"
       and _big._resume_cache_device(768, 768, 56) == "cpu")
    _devmod.plannable_free_vram = lambda: 12.0
    ck("12 GB free: a 56-frame cache (5.1 GB) is still parked on the CPU — above the 2.5 GB GPU cap (paging, 4 Sep)",
       _big._resume_cache_device(768, 768, 56) == "cpu")
    _devmod.plannable_free_vram = lambda: 4.5
    ck("4.5 GB free but the previous same-key 22-frame cache (1.85 GB) is on the GPU: reclaimable, stays on the GPU",
       _big._resume_cache_device(768, 640, 22, reclaim_gb=_big.resume_cache_gb(768, 640, 22)) == "cuda"
       and _big._resume_cache_device(768, 640, 22) == "cpu")
    _big._cache_device = "cpu"
    ck("a CPU-tier load never records on the GPU", _big._resume_cache_device(512, 416, 22) == "cpu")
finally:
    _devmod.plannable_free_vram = _free_orig

# --- load strength (slider × scale) and the dialled Turbo (steps / strength / off) ---------------
_sc = _SS.default_h3()
_sc.primary_scale = 0.8
_sc.blocks["h3blk_2"].primary_strength = 0.5
_eng.apply_state(_sc)
_m = {m.lora_name: m.multiplier for m in net.unet_loras}
ck("load strength: a block at 1.0 runs at the scale, a block at 0.5 at half of it",
   abs(_m["lora_unet_blocks_0_attn_qkv_proj"] - 0.8) < 1e-9 and abs(_m["lora_unet_blocks_2_mlp_fc1"] - 0.4) < 1e-9
   and abs(_m["lora_unet_token_refiner_blocks_1_mlp_fc1"] - 0.8) < 1e-9, _m)
_lat_s, _ = _eng.render_latent(_sc, frames=5, steps=2)
_sc1 = _sc.copy(); _sc1.primary_scale = 1.0
_lat_1, _ = _eng.render_latent(_sc1, frames=5, steps=2)
ck("...and the render moves with it", not torch.equal(_lat_s, _lat_1))
ck("SliderState carries the scales through copy / json",
   _sc.copy().primary_scale == 0.8 and _SS.from_json(_sc.to_json()).primary_scale == 0.8
   and _SS.from_json({}).primary_scale == 1.0)

_eng._turbo_net, _eng._turbo_adaln = net, []
_eng.REGIMES = _big.REGIMES = _H3E.REGIMES
ck("regime_params: presets", _eng.__class__.__dict__.get("regime_params") is None
   and _H3E.regime_params(_eng, "dial") == (4, 1.0) and _H3E.regime_params(_eng, "confirm") == (6, 0.75))
ck("regime_params: dialled numbers override, 0 = Turbo off",
   _H3E.regime_params(_eng, "dial", 3, 0.6) == (3, 0.6) and _H3E.regime_params(_eng, "confirm", 8, 0.0) == (8, 0.0))
_eng._turbo_net = None
ck("regime_params without a Turbo LoRA: steps still yours, strength None",
   _H3E.regime_params(_eng, "dial", 12) == (12, None))
_eng._turbo_net = net
_H3E.set_turbo_strength(_eng, 0.0)
ck("Turbo at 0: every Turbo module disabled", all(not m.enabled for m in net.unet_loras))
_H3E.set_turbo_strength(_eng, 0.6)
ck("Turbo at 0.6: modules back on at 0.6",
   all(m.enabled and abs(m.multiplier - 0.6) < 1e-9 for m in net.unet_loras))
_H3E.set_turbo_strength(_eng, 1.0)
_eng._turbo_net = None

_big.primary_network, _big.primary_hash, _big.donor_hash = net, "abc", None
_big.keyframe_signature = _H3E.keyframe_signature
_big.regime_params = _H3E.regime_params.__get__(_big)
_big.cache_key_for = _H3E.cache_key_for.__get__(_big)
_big._turbo_net, _big._steps = net, 6
_k1 = _big.cache_key_for(_sc1, frames=22, regime="dial")
_k2 = _big.cache_key_for(_sc, frames=22, regime="dial")
_k3 = _big.cache_key_for(_sc1, frames=22, regime="dial", steps=3, turbo_strength=1.0)
_k4 = _big.cache_key_for(_sc1, frames=22, regime="dial", steps=4, turbo_strength=1.0)
ck("render-cache setup key: changes with the load strength and with dialled steps, not with the preset spelled out",
   _k1 != _k2 and _k1 != _k3 and _k1 == _k4, (_k1, _k2, _k3, _k4))

# --- the LoKR whitelist fix: quantized Linear subclasses are mappable -----------------------------
from fizgig.networks.lora import _build_dit_linear_map  # noqa: E402


class ConvRotInt8Linear(nn.Linear):     # same class NAME the real int8 base uses
    pass


class _FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = ConvRotInt8Linear(8, 8)


class _FakeDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_FakeBlock()])


_map = _build_dit_linear_map(_FakeDiT(), None)
ck("LyCORIS module map includes quantized Linear subclasses (the H3 LoKR gap)",
   "lora_unet_blocks_0_qkv_proj" in _map)

# --- prompt disk cache: a hit never touches the TE ------------------------------------------------
from fizgig.repair_studio.h3_engine import H3RepairEngine  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

eng = H3RepairEngine()
eng.te_path = "X:/fake_te.safetensors"
eng._te_cache_dir = os.path.join(TMP, "te_prompts")
os.makedirs(eng._te_cache_dir, exist_ok=True)
_prompt = "a portrait of zwxem"
_emb = torch.randn(7, 5120)
save_file({"hidden_states": _emb}, eng._prompt_disk_path(_prompt))

import fizgig.minimax.sampling as _samp  # noqa: E402
_real_esp = _samp.encode_sample_prompts
_samp.encode_sample_prompts = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("TE loaded!"))
try:
    got = eng._encode_prompt(_prompt)
    ck("disk-cache hit bypasses the 32B TE entirely",
       got.shape == (1, 7, 5120) and torch.equal(got[0], _emb))
    got2 = eng._encode_prompt(_prompt)      # in-memory hit
    ck("...and the session cache serves repeats", got2 is got)
finally:
    _samp.encode_sample_prompts = _real_esp

# --- GUI: Repair Studio tri-state (headless) ------------------------------------------------------
import tkinter as tk  # noqa: E402
import lora_trainer_gui as G  # noqa: E402

G.LoRATrainerGUI.save_prefs = lambda self, *a, **k: None      # neuter; never patch _persist_disabled
G.LoRATrainerGUI._save_training_queue = lambda self, *a, **k: None

PREFS = os.path.join(os.path.dirname(__file__), "..", "prefs.json")
_before = hashlib.sha256(open(PREFS, "rb").read()).hexdigest() if os.path.exists(PREFS) else None

root = tk.Tk()
app = G.LoRATrainerGUI(root)
app.prefs["minimax_extras_prompt_dismissed"] = True

app.repair_family_var.set("minimax")
app._on_repair_family_changed()
root.update()
ck("H3 slider panel builds 52 rows", len(app.repair_block_vars) == 52
   and "h3blk_49" in app.repair_block_vars and "h3_rf_1" in app.repair_block_vars)
ck("...master controls hidden (no block map)",
   app._repair_master_container.winfo_manager() == "")
ck("...DiT radios disabled (H3 auto-plans)",
   "disabled" in app._repair_dit_radio_a.state())
ck("...preview res defaults to 768", app.repair_res_var.get() == "768")
ck("...presets collapse to Reset All",
   app._repair_preset_list()[0].endswith("Reset All"))
ck("...state is H3-shaped", len(app.repair_state.blocks) == 52)

_routed = []
app._repair_engine_plan_h3 = lambda: _routed.append("h3") or {}
app._repair_engine_plan()
ck("engine dispatch routes minimax to the H3 engine", _routed == ["h3"])

app.repair_family_var.set("klein")
app._on_repair_family_changed()
root.update()
ck("switching back to Klein restores 32 rows + master controls",
   len(app.repair_block_vars) == 32
   and app._repair_master_container.winfo_manager() != ""
   and "disabled" not in app._repair_dit_radio_a.state())

root.destroy()

_after = hashlib.sha256(open(PREFS, "rb").read()).hexdigest() if os.path.exists(PREFS) else None
ck("prefs.json is byte-identical after the run", _before == _after)

print()

# --- ref2va: references ride as condition rows with the prompt's token tags ----------------------
from PIL import Image as _PILImage  # noqa: E402
_eng_r = _MiniEngine()
_eng_r._prompt_cache_tags = None
_L = _txt.shape[1]
_tags = torch.zeros(_L, dtype=torch.long); _tags[:2] = 1        # 2 "video" rows up front, say
_calls = []
def _enc(prompt, images=None):
    _calls.append((prompt, len(images) if images else 0))
    if images:
        _eng_r._prompt_cache_tags = _tags
    return _txt
_eng_r._encode_prompt = _enc
_st_r = _SS.default_h3()
_st_r.seed, _st_r.prompt, _st_r.preview_width, _st_r.preview_height, _st_r.preview_frames = 3, "x", 64, 64, 5
_ref_img = _PILImage.new("RGB", (64, 64), (200, 100, 50))
_ref_lat = torch.randn(1, 24, 1, 4, 4)
_st_r.references = [(_ref_img, _ref_lat)]
_lat_ref, _ = _eng_r.render_latent(_st_r, frames=5, steps=2)
_st_r.references = None
_lat_plain, _ = _eng_r.render_latent(_st_r, frames=5, steps=2)
ck("a reference changes the render and the prompt was encoded WITH the picture",
   not torch.equal(_lat_ref, _lat_plain) and _calls[0] == ("x", 1) and _calls[1] == ("x", 0), _calls)
_st_r.references = [(_ref_img, _ref_lat)]
_sig = _H3E.keyframe_signature(_st_r)
ck("the conditioning signature covers references", len(_sig) == 1 and _sig[0][0] == "ref")
_fp1 = _H3E._ref_fingerprint([_ref_img]); _fp2 = _H3E._ref_fingerprint([_PILImage.new("RGB", (64, 64), (0, 0, 0))])
ck("reference fingerprints differ per picture, empty without", _fp1 != _fp2 and _H3E._ref_fingerprint([]) == "")
_eng_r._te_cache_dir, _eng_r.te_path = "C:/tmp/tecache", "te.safetensors"
ck("the prompt disk key differs with references", _H3E._prompt_disk_path(_eng_r, "x") != _H3E._prompt_disk_path(_eng_r, "x", _fp1))
_big.dit_path = "a.safetensors"
_k_a = _big.cache_key_for(_sc1, frames=22, regime="dial")
_big.dit_path = "b.safetensors"
_k_b = _big.cache_key_for(_sc1, frames=22, regime="dial")
ck("render-cache setup key changes with the H3 checkpoint (fl2va vs ref2va)", _k_a != _k_b)


# --- the text encoder parked in system RAM: decision logic with stubs -----------------------------
class _FakeTE:
    def __init__(self, vision):
        self.vision = vision
        self.moves = []
        class _LS:
            def __init__(s2): s2.unloaded = 0
            def unload_all(s2): s2.unloaded += 1
        self.layer_streamer = _LS()
        class _Child(torch.nn.Module):
            def __init__(s2, owner, nm, dev="cuda:0"):
                super().__init__(); s2.owner, s2.nm = owner, nm
                s2.w = torch.nn.Parameter(torch.zeros(1)); s2.home = dev
            def parameters(s2, recurse=True):
                class _P:                          # reports the pretend home device
                    device = s2.home
                yield _P()
            def to(s2, dev, *a, **k): s2.owner.moves.append((s2.nm, str(dev))); return s2
        class _Decoder(torch.nn.Module):
            def __init__(s2, owner):
                super().__init__()
                s2.embed_tokens = _Child(owner, "embed", "cpu"); s2.norm = _Child(owner, "norm")   # text-only: embeddings live on the CPU
                s2.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        class _Model(torch.nn.Module):
            def __init__(s2, owner, vision):
                super().__init__()
                s2.language_model = _Decoder(owner)
                if vision: s2.visual = _Child(owner, "visual")
        self.model = _Model(self, vision)
    def _get_decoder_and_embeddings(self):
        return self.model.language_model, self.model.language_model.embed_tokens

_loads = []
_pe = _MiniEngine()
_pe._te_parked, _pe._te_parked_vision, _pe.te_path = None, False, "te.safetensors"
for _nm in ("_te_get", "_te_park", "_te_wake", "_te_free"):
    setattr(_pe, _nm, getattr(_H3E, _nm).__get__(_pe))
_pe._te_resident_modules = _H3E._te_resident_modules
_pe._TE_PARK_MIN_AVAIL_GB, _pe._TE_KEEP_MIN_AVAIL_GB = _H3E._TE_PARK_MIN_AVAIL_GB, _H3E._TE_KEEP_MIN_AVAIL_GB
import fizgig.minimax.embedderH2D as _h2d, fizgig.minimax.embedder as _emb  # noqa: E402
_orig_h2d, _orig_planned = _h2d.load_minimax_h3_te, _emb.load_minimax_h3_te_planned
_h2d.load_minimax_h3_te = lambda path, **kw: (_loads.append(("stream", kw.get("with_vision"))) or _FakeTE(kw.get("with_vision")))
_emb.load_minimax_h3_te_planned = lambda path, **kw: (_loads.append(("resident", kw.get("with_vision"))) or _FakeTE(kw.get("with_vision")))
try:
    _pe._ram_available_gb = lambda: 4.0
    _te, _keep = _pe._te_get(False)
    ck("RAM short (4 GB): the resident build, not kept", _loads == [("resident", False)] and not _keep and _pe._te_parked is None)
    _loads.clear()
    _pe._ram_available_gb = lambda: 100.0
    _te, _keep = _pe._te_get(False)
    ck("RAM plentiful: the streamed build (always with vision), parked", _loads == [("stream", True)] and _keep and _pe._te_parked is _te)
    _pe._te_park()
    ck("park: rings unloaded, resident parts to the CPU (not the layers)",
       _te.layer_streamer.unloaded == 1 and set(_te.moves) == {("embed", "cpu"), ("norm", "cpu"), ("visual", "cpu")})
    _te.moves.clear(); _loads.clear()
    _pe.device = "cuda"
    _te2, _keep2 = _pe._te_get(False)
    ck("next prompt: the parked encoder, woken (no load, each part back to where it lived — the CPU embedding stays on the CPU)",
       _te2 is _te and _keep2 and not _loads and set(_te.moves) == {("embed", "cpu"), ("norm", "cuda:0"), ("visual", "cuda:0")}, _te.moves)
    _te3, _keep3 = _pe._te_get(True)
    ck("references arrive: the same parked build serves them (no second build in the session)",
       not _loads and _te3 is _te and _pe._te_parked_vision)
    _te4, _ = _pe._te_get(False)
    ck("...and the vision build serves a plain prompt afterwards", _te4 is _te3 and not _loads)
    _pe._te_park()
    ck("park moves the vision tower too", ("visual", "cpu") in _te3.moves)
    _pe._ram_available_gb = lambda: 5.0
    _pe._te_park()
    ck("RAM gets short while parked: the encoder is released", _pe._te_parked is None)
finally:
    _h2d.load_minimax_h3_te, _emb.load_minimax_h3_te_planned = _orig_h2d, _orig_planned


# --- small-RAM machine: the resident encoder gets the card to itself (DiT parked FIRST) ----------
_ev = []
class _DitStub:
    def to(self, dev, *a, **k): _ev.append(("dit", str(dev))); return self
class _TEStub:
    layer_streamer = None
    def encode(self, prompt, max_length=None): _ev.append(("encode",)); return torch.zeros(1, 4, 8)
    def encode_with_reference(self, prompt, images): _ev.append(("encode_ref",)); return torch.zeros(1, 4, 8), torch.zeros(4, dtype=torch.long)
_le = _MiniEngine()
for _nm in ("_encode_prompt", "_te_get", "_te_can_park", "_te_park", "_te_wake", "_te_free", "_status", "_prompt_disk_path", "_ram_available_gb"):
    setattr(_le, _nm, getattr(_H3E, _nm).__get__(_le))
_le._te_resident_modules, _le._ref_fingerprint = _H3E._te_resident_modules, _H3E._ref_fingerprint
_le._TE_PARK_MIN_AVAIL_GB, _le._TE_KEEP_MIN_AVAIL_GB = _H3E._TE_PARK_MIN_AVAIL_GB, _H3E._TE_KEEP_MIN_AVAIL_GB
_le._te_parked, _le._te_parked_vision, _le.te_path, _le._te_cache_dir = None, False, "te.safetensors", None
_le._prompt_cache_key = _le._prompt_cache = _le._prompt_cache_tags = None
_le.on_status = None; _le.dit = _DitStub(); _le.device = "cuda"; _le._turbo_net = None
_orig_planned2 = _emb.load_minimax_h3_te_planned
_emb.load_minimax_h3_te_planned = lambda path, **kw: (_ev.append(("load_resident",)) or _TEStub())
_orig_free = _devmod.plannable_free_vram
try:
    _devmod.plannable_free_vram = lambda: 9.0             # the DiT is on the card
    _le._ram_available_gb = lambda: 10.0                  # ...and the box has no RAM to park
    _le._encode_prompt("x")
    ck("10 GB RAM: DiT parked BEFORE the resident encoder loads, encoder freed, DiT restored",
       _ev == [("dit", "cpu"), ("load_resident",), ("encode",), ("dit", "cuda")] and _le._te_parked is None, _ev)
finally:
    _emb.load_minimax_h3_te_planned = _orig_planned2
    _devmod.plannable_free_vram = _orig_free

if _fails:
    print(f"{len(_fails)} FAILED: " + ", ".join(_fails))
    sys.exit(1)

# --- the AdaLN injection is composed from the built-in Turbo AND a Turbo LoRA loaded as primary -----
class _FakeMod:
    pass


_m0, _m2, _mt = _FakeMod(), _FakeMod(), _FakeMod()
_A, _B = torch.ones(2, 4), torch.ones(6, 2)
_ce = _MiniEngine()
_ce._turbo_adaln = [(_mt, _A, _B)]                  # built-in rows, unscaled
_ce._turbo_strength, _ce._turbo_load_strength = 1.0, 0.75
_ce._primary_adaln = [("lora_unet_blocks_0_adaln_linear", _m0, _A, _B),
                      ("lora_unet_blocks_2_adaln_linear", _m2, _A, _B)]
_ce._donor_adaln = []
_mf = _FakeMod()
_ce._primary_adaln.append(("lora_unet_final_layer_adaln_proj_linear", _mf, _A, _B))   # no block owns it
_stc = _SS.default_h3()
_stc.primary_scale = 0.8
_stc.blocks["h3blk_2"].primary_strength = 0.5
_ce._last_state = _stc
_pairs, _sig = _ce._adaln_pairs_now()
_fac = {id(m): float(b[0, 0]) for m, a, b in _pairs}
ck("AdaLN rows: built-in at the dialled 1.0, primary block 0 at 0.8, block 2 at 0.8x0.5",
   abs(_fac[id(_mt)] - 1.0) < 1e-6 and abs(_fac[id(_m0)] - 0.8) < 1e-6 and abs(_fac[id(_m2)] - 0.4) < 1e-6, _fac)
_stc.blocks["h3blk_0"].primary_enabled = False
_pairs, _ = _ce._adaln_pairs_now()
ck("...a block switched off drops its AdaLN row", id(_m0) not in {id(m) for m, a, b in _pairs})
ck("...the final-layer row (no block owns it) rides at the load strength while any block is on",
   abs({id(m): float(b[0, 0]) for m, a, b in _pairs}[id(_mf)] - 0.8) < 1e-6)
for _row in _stc.blocks.values():
    _row.primary_enabled = False
_pairs, _ = _ce._adaln_pairs_now()
ck("...and goes with every block off (the LoRA is out)", id(_mf) not in {id(m) for m, a, b in _pairs}
   and id(_m2) not in {id(m) for m, a, b in _pairs})
for _row in _stc.blocks.values():
    _row.primary_enabled = True
_stc.blocks["h3blk_0"].primary_enabled = False
_ce._turbo_strength = 0.0
_pairs, _ = _ce._adaln_pairs_now()
ck("...Turbo at 0 drops the built-in rows, keeps the primary's", {id(m) for m, a, b in _pairs} == {id(_m2), id(_mf)})
_pairs, _ = _ce._adaln_pairs_now(no_lora=True)
ck("...a no-LoRA render installs nothing (Turbo 0)", not _pairs)
_ce._turbo_strength = 0.6
_pairs, _ = _ce._adaln_pairs_now(no_lora=True)
ck("...a no-LoRA render keeps only the built-in Turbo rows", {id(m) for m, a, b in _pairs} == {id(_mt)}
   and abs({id(m): float(b[0, 0]) for m, a, b in _pairs}[id(_mt)] - 0.6) < 1e-6)
ck("_collect_adaln_pairs: a LoRA without AdaLN keys yields none on the tiny DiT",
   _H3E.__module__ and __import__("fizgig.repair_studio.h3_engine", fromlist=["_collect_adaln_pairs"])._collect_adaln_pairs(dit, sd) == [])


# --- _apply_lora prefilters modules to the base's Linears (a Turbo LoRA's AdaLN rows are not modules) -
_sd2 = dict(sd)
_bad = "lora_unet_blocks_1_attn_qkv_proj"          # exists, but with the WRONG in_features
_sd2[f"{_bad}.lora_down.weight"] = torch.randn(4, 999) * 0.1
_sd2[f"{_bad}.lora_up.weight"] = torch.randn(dict(dit.named_modules())["blocks.1.attn.qkv_proj"].out_features, 4) * 0.1
_sd2[f"{_bad}.alpha"] = torch.tensor(4.0)
_sd2["lora_unet_nowhere_linear.lora_down.weight"] = torch.randn(4, 8)
_sd2["lora_unet_nowhere_linear.lora_up.weight"] = torch.randn(8, 4)
_net2 = _apply_lora(dit, _sd2, 1.0, "cpu", torch.float32)
ck("mismatched / unknown modules are dropped, the 3 that fit are wired",
   len(_net2.unet_loras) == 3 and _bad not in {m.lora_name for m in _net2.unet_loras})
try:
    _apply_lora(dit, {"lora_unet_nowhere_linear.lora_down.weight": torch.randn(4, 8),
                      "lora_unet_nowhere_linear.lora_up.weight": torch.randn(8, 4)}, 1.0, "cpu", torch.float32)
    ck("a LoRA with nothing that fits is refused", False)
except RuntimeError:
    ck("a LoRA with nothing that fits is refused", True)

print("ALL PASS")
