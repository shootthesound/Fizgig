"""MOSS-Image driver tests (T1, T5, T10-T12): CPU, tiny random weights, the reference pipeline as the yardstick."""
import math
import os
import subprocess
import sys

import pytest
import torch

from moss_tiny import TINY, pair, rel

MODULES = ("proj.q", "proj.k", "proj.v", "proj.o", "proj.ffn_up_gate", "proj.ffn_down")
RESULTS = {}


def _desc():
    from fizgig.families import registry
    return registry.get("moss_image")


def _drv():
    return _desc().load_driver()


# ---- T1: description, block map, LoRA keys --------------------------------------------------------------------------
def test_t1_description_and_registry():
    from fizgig.families import registry
    d = _desc()
    assert d.validate() == [] and not d.hidden and registry.by_arch_id("mossimage") is d
    assert d in registry.training_families()                # visible since M4
    assert d.n_blocks == 42 and d.lora.block_modules == MODULES and d.latent_channels == 128
    assert {f.role for f in d.model_files} >= {"dit", "vae", "text_encoder"}
    keys = [o.key for o in d.options]
    assert "MOSS_CAPTION_DROPOUT" in keys
    cd = next(o for o in d.options if o.key == "MOSS_CAPTION_DROPOUT")
    assert cd.kind == "entry" and cd.default == "0" and cd.tokens == "caption_dropout={}"


def test_t1_block_map_252_and_no_text_refiner():
    drv = _drv()
    bm = drv.block_map()
    mods = [m for g in bm for b in g.blocks for m in b.modules]
    assert len(bm[0].blocks) == 42 and len(mods) == 252 == len(set(mods))
    assert not any("text_embedder" in m or "modulation" in m for m in mods)
    fz, _ = pair()
    small = [m for g in drv.block_map(fz) for b in g.blocks for m in b.modules]
    assert len(small) == 6 * TINY["num_layers"]
    names = dict(fz.named_modules())
    assert all(isinstance(names[m], torch.nn.Linear) for m in small)


def _kohya(block, module):
    stem = "lora_unet_layers_%d_%s" % (block, module.replace(".", "_"))
    return [f"{stem}.lora_down.weight", f"{stem}.lora_up.weight", f"{stem}.alpha"]


def _peft(block, module, prefix="diffusion_model."):
    stem = f"{prefix}layers.{block}.{module}"
    return [f"{stem}.lora_A.weight", f"{stem}.lora_B.weight"]


def test_t1_key_round_trip_252():
    from fizgig.families.lorafile import lora_pairs
    d = _desc()
    known = {m for g in _drv().block_map() for b in g.blocks for m in b.modules}
    k = lora_pairs(d, [x for b in range(42) for m in MODULES for x in _kohya(b, m)])
    assert len(k) == 252 and {p[0] for p in k} == known and all(p[3] for p in k)
    for prefix in ("diffusion_model.", "transformer.", ""):
        p = lora_pairs(d, [x for b in range(42) for m in MODULES for x in _peft(b, m, prefix)])
        assert {q[0] for q in p} == known, prefix


def _fake(path, keys):
    from safetensors.torch import save_file
    save_file({k: torch.zeros(1) for k in keys}, str(path))
    return str(path)


def test_t1_family_of_lora(tmp_path, monkeypatch):
    from fizgig.families import registry
    d = _desc()
    real = registry.training_families
    monkeypatch.setattr(registry, "training_families", lambda: real() + [d])
    moss = _fake(tmp_path / "m.safetensors", [x for b in range(42) for m in MODULES for x in _kohya(b, m)])
    assert registry.family_of_lora(moss) is d
    moss_peft = _fake(tmp_path / "mp.safetensors", [x for m in MODULES for x in _peft(3, m)])
    assert registry.family_of_lora(moss_peft) is d
    # other families' files stay theirs
    qwen = _fake(tmp_path / "q.safetensors",
                 [f"lora_unet_transformer_blocks_2_attn_to_q.{s}" for s in ("lora_down.weight", "lora_up.weight")])
    assert registry.family_of_lora(qwen) is not d and registry.family_of_lora(qwen) is not None
    anima = _fake(tmp_path / "a.safetensors",
                  [f"lora_unet_blocks_0_self_attn_q_proj.{s}" for s in ("lora_down.weight", "lora_up.weight")])
    assert registry.family_of_lora(anima) is not d
    f3 = _fake(tmp_path / "f.safetensors",
               [f"lora_unet_single_blocks_3_q_proj.{s}" for s in ("lora_down.weight", "lora_up.weight")])
    assert registry.family_of_lora(f3) is not d


def test_t1_lora_save_reload(tmp_path):
    from safetensors import safe_open

    from fizgig.families.lora import FamilyLoRA
    drv = _drv()
    fz, _ = pair()
    net = FamilyLoRA(fz.requires_grad_(False), drv)
    net.add_trainable(4, 4.0)
    assert len(net.trainable_modules()) == 6 * TINY["num_layers"]
    for p in net.parameters():
        p.data.normal_(0, 0.1)
    path = str(tmp_path / "l.safetensors")
    net.save(path)
    with safe_open(path, "pt") as f:
        keys = set(f.keys())
    assert "lora_unet_layers_3_proj_ffn_up_gate.lora_down.weight" in keys
    assert "lora_unet_layers_0_proj_q.alpha" in keys
    assert len(keys) == 3 * 6 * TINY["num_layers"] and not any("text_embedder" in k for k in keys)
    fz2, _ = pair()
    net2 = FamilyLoRA(fz2.requires_grad_(False), drv)
    net2.add_trainable(4, 4.0)
    assert net2.load_trainable(path) == 6 * TINY["num_layers"]
    x, c = torch.randn(1, TINY["latent_channels"], 2, 3), torch.randn(1, 5, TINY["text_dim"])
    base, _ = pair()
    with torch.no_grad():
        a, b, o = fz(x, 100.0, c), fz2(x, 100.0, c), base(x, 100.0, c)
    # the file holds bf16 factors: the reload equals the trained adapter to bf16 rounding of its effect
    assert (a - b).abs().max() < 0.02 * (a - o).abs().max()


# ---- T5: loss parity, LoRA gradients ------------------------------------------------------------------------------
class _RefLoRA(torch.nn.Module):
    def __init__(self, base, A, B, scale):
        super().__init__()
        self.base, self.scale = base, scale
        self.A, self.B = torch.nn.Parameter(A.clone()), torch.nn.Parameter(B.clone())

    @property
    def weight(self):                      # the reference reads proj['q'].weight.dtype
        return self.base.weight

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T


@pytest.mark.parametrize("gc", [False, True])
def test_t5_loss_and_lora_grads_equal_reference(gc):
    from fizgig.families.lora import FamilyLoRA
    from fizgig.moss_image import sampling as S
    drv = _drv()
    drv.set_options({})
    fz, ref = pair(seed=21)
    fz.requires_grad_(False)
    ref.requires_grad_(False)
    net = FamilyLoRA(fz, drv)
    net.add_trainable(4, 2.0)
    g0 = torch.Generator().manual_seed(9)
    for p in net.parameters():
        p.data.copy_(torch.randn(p.shape, generator=g0) * 0.1)
    fz.enable_gradient_checkpointing(gc)
    # the same adapters on the reference model's proj.q / proj.ffn_down, as plain autograd
    hooks = {}
    for name, w in net.wrapped.items():
        parent, leaf = name.rsplit(".", 1)
        A, B = w.adapters["lora"][0].weight.data, w.adapters["lora"][1].weight.data
        mod = _RefLoRA(ref.get_submodule(name), A, B, w.scales["lora"])
        setattr(ref.get_submodule(parent), leaf, mod)
        hooks[name] = mod
    x0 = torch.randn(1, TINY["latent_channels"], 3, 4)
    c = torch.randn(1, 7, TINY["text_dim"])
    gen = torch.Generator().manual_seed(3)
    loss, info = drv.training_loss(fz, x0, {"hidden_states": c}, gen)
    loss.backward()
    # replay the driver's draws: noise first, then sigma
    g = torch.Generator().manual_seed(3)
    eps = torch.randn(x0.shape, generator=g)
    sigma = S.sample_t(g)
    assert sigma == info["t"]
    xt = (1 - sigma) * x0 + sigma * eps
    want = torch.nn.functional.mse_loss(ref(xt, torch.tensor([1000 * sigma]), c).sample.float(), eps - x0)
    want.backward()
    RESULTS[f"T5 loss gc={gc}"] = abs(loss.item() - want.item()) / want.item()
    assert abs(loss.item() - want.item()) <= 1e-6 * want.item()
    worst = 0.0
    for name, w in net.wrapped.items():
        for got, exp in ((w.adapters["lora"][0].weight.grad, hooks[name].A.grad),
                         (w.adapters["lora"][1].weight.grad, hooks[name].B.grad)):
            assert got is not None and float(got.abs().sum()) > 0
            worst = max(worst, rel(got, exp))
    RESULTS[f"T5 LoRA grads gc={gc} (worst rel)"] = worst
    assert worst <= 1e-5


def test_t5_masked_cond_and_caption_dropout(tmp_path):
    """A padded cached prompt trains as the unpadded one; caption dropout swaps in the cached empty prompt at the
    configured share, and 0 (the default) never draws from the generator."""
    from safetensors.torch import save_file
    drv = _drv()
    fz, _ = pair(seed=22)
    x0 = torch.randn(1, TINY["latent_channels"], 2, 2)
    c = torch.randn(1, 4, TINY["text_dim"])
    padded = drv.pad_conditioning([{"hidden_states": c[0]}, {"hidden_states": torch.randn(9, TINY["text_dim"])}])[0]
    drv.set_options({})
    with torch.no_grad():
        a, _ = drv.training_loss(fz, x0, {"hidden_states": c}, torch.Generator().manual_seed(1))
        b, _ = drv.training_loss(fz, x0, {k: v[None] for k, v in padded.items()}, torch.Generator().manual_seed(1))
    assert rel(b, a) <= 1e-6
    # dropout: the uncond file the text stage writes, read by prepare_training
    unc = torch.randn(3, TINY["text_dim"]).to(torch.bfloat16)
    save_file({"hidden_states": unc}, str(tmp_path / "mossimage_empty_prompt.safetensors"))
    group = type("G", (), {"datasets": [type("D", (), {"cache_directory": str(tmp_path)})()]})()
    drv.set_options({"caption_dropout": "1"})
    drv.prepare_training(fz, group)
    with torch.no_grad():
        d1, _ = drv.training_loss(fz, x0, {"hidden_states": c}, torch.Generator().manual_seed(1))
        g = torch.Generator().manual_seed(1)
        eps = torch.randn(x0.shape, generator=g)
        from fizgig.moss_image import sampling as S
        s = S.sample_t(g)
        xt = (1 - s) * x0 + s * eps
        want = torch.nn.functional.mse_loss(fz(xt, torch.tensor([1000 * s]), unc[None].float()), eps - x0)
    assert rel(d1, want) <= 1e-6
    drv.set_options({"caption_dropout": "0.25"})
    drv.prepare_training(fz, group)
    hits = sum(drv._text({"hidden_states": c}, "cpu", torch.Generator().manual_seed(i))[0].shape[1] == 3
               for i in range(2000))
    assert 400 < hits < 600
    with pytest.raises(ValueError):
        drv.set_options({"caption_dropout": "1.5"})


def test_t5_generate_matches_sampler_and_travel():
    from fizgig.moss_image import sampling as S
    drv = _drv()
    drv.set_options({})
    fz, _ = pair(seed=23)
    fz.requires_grad_(False)
    cond = {"hidden_states": torch.randn(1, 5, TINY["text_dim"])}
    neg = {"hidden_states": torch.randn(1, 2, TINY["text_dim"])}
    seen = []
    lat = drv.generate(fz, cond, 64, 48, steps=4, seed=3, cfg=3.0, neg_cond=neg,
                       options=(("uncond_skip_layers", (1,)),), on_step=lambda i, n: seen.append(i))
    want = S.sample(fz, cond["hidden_states"], 48, 64, steps=4, seed=3, cfg=3.0, neg=neg["hidden_states"],
                    uncond_skip_layers=(1,))
    assert torch.equal(lat, want) and lat.shape == (1, TINY["latent_channels"], 3, 4) and seen == [0, 1, 2, 3, 4]
    p = drv.pad_conditioning([{"hidden_states": cond["hidden_states"][0]}, {"hidden_states": torch.randn(8, 48)}])
    lat2 = drv.generate(fz, {k: v[None] for k, v in p[0].items()}, 64, 48, steps=4, seed=3, cfg=3.0, neg_cond=neg,
                        options=(("uncond_skip_layers", (1,)),))
    RESULTS["T5 travel padded vs unpadded"] = rel(lat2, lat)
    assert rel(lat2, lat) <= 1e-5
    with pytest.raises(ValueError):
        drv.generate(fz, cond, 64, 48, steps=2, seed=1, cfg=3.0)


# ---- T11: INT8 on CPU ---------------------------------------------------------------------------------------------
def test_t11_int8_base_trains_on_cpu():
    from fizgig.families import quant
    from fizgig.families.lora import FamilyLoRA
    drv = _drv()
    drv.set_options({})
    fz, _ = pair(seed=24)
    fz = fz.to(torch.bfloat16).requires_grad_(False)
    x0 = torch.randn(1, TINY["latent_channels"], 2, 3)
    c = torch.randn(1, 5, TINY["text_dim"])
    with torch.no_grad():
        base = fz(x0, 500.0, c).float()
    try:
        n = quant.quantize(fz, drv, "int8", "cpu")
    except Exception as e:                                 # noqa: BLE001
        pytest.skip(f"INT8 path unavailable on CPU: {e}")
    assert n == 6 * TINY["num_layers"]
    try:
        with torch.no_grad():
            q = fz(x0, 500.0, c).float()
    except RuntimeError as e:
        pytest.skip(f"torch._int_mm has no CPU kernel here: {e}")
    RESULTS["T11 int8 vs bf16 forward"] = rel(q, base)
    assert rel(q, base) < 0.05
    net = FamilyLoRA(fz, drv)
    net.add_trainable(4, 4.0)
    loss, _ = drv.training_loss(fz, x0.to(torch.bfloat16), {"hidden_states": c}, torch.Generator().manual_seed(0))
    loss.backward()
    assert math.isfinite(loss.item()) and all(p.grad is not None for p in net.parameters()[1::2])


# ---- T12: block-swap bookkeeping with skipped blocks --------------------------------------------------------------
def test_t12_skipped_blocks_keep_the_swap_schedule():
    fz, _ = pair(seed=25)
    calls = []

    class Fake:
        def wait_for_block(self, i):
            calls.append(("wait", i))

        def submit_move_blocks_forward(self, blocks, i):
            calls.append(("move", i))
    x, c = torch.randn(1, TINY["latent_channels"], 2, 2), torch.randn(1, 3, TINY["text_dim"])
    with torch.no_grad():
        plain = fz(x, 10.0, c, skip_layers=(1, 2))
        fz.offloader, fz.blocks_to_swap = Fake(), 2
        swapped = fz(x, 10.0, c, skip_layers=(1, 2))
    n = TINY["num_layers"]
    assert calls == [e for i in range(n) for e in (("wait", i), ("move", i))]
    assert torch.equal(plain, swapped)
    assert _drv().max_blocks_to_swap(fz) == n - 2 and _drv().max_blocks_to_swap() == 40


def test_text_cache_one_request_with_the_empty_prompt(tmp_path):
    """The text stage hands the worker a dataset's captions in one request (its layer-major pass), the empty prompt
    riding along when a cache folder still lacks it; outside a cache stage nothing is added."""
    from argparse import Namespace

    from safetensors.torch import load_file

    from fizgig.moss_image.driver import TEXT_CACHE_BATCH, UNCOND_FILE

    class TE:
        def __init__(self):
            self.calls = []

        def encode(self, caps):
            self.calls.append(list(caps))
            return [torch.full((len(c) + 3, 8), float(len(c))) for c in caps]

        def unload(self):
            self.calls.append("unload")

    ds = type("D", (), {"cache_directory": str(tmp_path / "c")})()
    drv, te = _drv(), TE()
    args = Namespace(batch_size=None)
    assert drv.cache_stage("text", [ds], args, "cpu", {}) is False
    assert args.batch_size == TEXT_CACHE_BATCH
    out = drv.encode_text(te, ["ab", "abcd"])
    assert [o["hidden_states"].shape[0] for o in out] == [5, 7]
    out = drv.encode_text(te, ["x"])                             # already have it: not asked again
    drv.unload_text_encoder(te)
    assert te.calls == [["ab", "abcd", ""], ["x"], "unload"]
    assert load_file(str(tmp_path / "c" / UNCOND_FILE))["hidden_states"].shape == (3, 8)
    te2 = TE()
    drv.cache_stage("text", [ds], Namespace(batch_size=4), "cpu", {})
    drv.encode_text(te2, ["ab"])                                 # the file exists now
    drv.unload_text_encoder(te2)
    assert te2.calls == [["ab"], "unload"]
    te3 = TE()
    drv.encode_text(te3, ["ab", ""])                             # previews / workbench: as asked
    assert te3.calls == [["ab", ""]]


# ---- T10: the CLI end to end (cache text + latents, train 2 epochs with previews, reload) ---------------------------
def test_t10_cli_cache_train_reload(tmp_path):
    import json

    import numpy as np
    from PIL import Image
    from safetensors.torch import load_file, save_file

    from fizgig.flux3.autoencoder import AutoEncoder
    from fizgig.moss_image import te_env
    from fizgig.moss_image.model import MossImageDiT
    try:
        py = te_env.ensure()
    except Exception as e:                                 # noqa: BLE001
        pytest.skip(f"side venv unavailable: {e}")
    tok = os.environ.get("FIZGIG_MOSS_TOKENIZER", "/tmp/moss/tokenizer")
    if not os.path.isfile(os.path.join(tok, "tokenizer.json")):
        pytest.skip("MOSS tokenizer unavailable")
    here = os.path.dirname(__file__)
    src = os.path.join(here, "..", "src")
    # tiny models: the TE (hidden 64), a DiT reading it, a random FLUX.2 VAE
    (tmp_path / "caps.json").write_text(json.dumps(["x"]))
    r = subprocess.run([py, os.path.join(here, "moss_te_tiny_build.py"), str(tmp_path / "te"), tok,
                        str(tmp_path / "caps.json")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    torch.manual_seed(0)
    dit = MossImageDiT(**dict(TINY, text_dim=64, latent_channels=128))
    save_file({k: v.contiguous() for k, v in dit.state_dict().items()}, str(tmp_path / "dit.safetensors"))
    ae = AutoEncoder()
    save_file({k: v.contiguous() for k, v in ae.state_dict().items()}, str(tmp_path / "vae.safetensors"))
    img_dir, cache = tmp_path / "imgs", tmp_path / "cache"
    img_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(img_dir / f"p{i}.png")
        (img_dir / f"p{i}.txt").write_text(f"a photo number {i} of sks")
    toml = tmp_path / "ds.toml"
    toml.write_text(f'[general]\nresolution = [64, 64]\ncaption_extension = ".txt"\nbatch_size = 1\n'
                    f'enable_bucket = true\nbucket_no_upscale = true\n\n[[datasets]]\n'
                    f'image_directory = "{img_dir.as_posix()}"\ncache_directory = "{cache.as_posix()}"\n')
    env = dict(os.environ, PYTHONPATH=os.path.abspath(src), FIZGIG_MOSS_TOKENIZER=tok)
    env["FIZGIG_MOSS_TE_MODE"] = "stream"           # the layer-major path, as on a 12 GB card
    te_dir = str(tmp_path / "te" / "text_encoder")
    # the worker finds the tokenizer beside the text_encoder folder, as in the official repo layout
    import shutil
    shutil.copytree(tok, str(tmp_path / "te" / "tokenizer"))       # (a symlink needs admin rights on Windows)
    run = lambda *a: subprocess.run([sys.executable, *a], capture_output=True, text=True, env=env, cwd=str(tmp_path))
    r = run("-m", "fizgig.families.cache", "--family", "moss_image", "--stage", "latents", "--dataset_config",
            str(toml), "--model", str(tmp_path / "vae.safetensors"), "--device", "cpu")
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    r = run("-m", "fizgig.families.cache", "--family", "moss_image", "--stage", "text", "--dataset_config",
            str(toml), "--model", te_dir, "--device", "cpu")
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    assert (cache / "mossimage_empty_prompt.safetensors").is_file()
    log = r.stdout + r.stderr
    assert "layer-major" in log and "4 captions" in log, log[-3000:]   # 3 captions + the empty prompt, one pass
    lat = [f for f in os.listdir(cache) if f.endswith("_mossimage.safetensors")]
    assert len(lat) >= 3, os.listdir(cache)
    if torch.cuda.is_available():
        (tmp_path / "prompts.txt").write_text("a photo of sks\n")
        out = tmp_path / "out"
        r = run("-m", "fizgig.families.train", "--family", "moss_image", "--dit", str(tmp_path / "dit.safetensors"),
                "--dataset_config", str(toml), "--output_dir", str(out), "--output_name", "tiny", "--precision",
                "bf16", "--network_dim", "4", "--network_alpha", "4", "--max_train_epochs", "2", "--learning_rate",
                "1e-3", "--optimizer_type", "adamw", "--compile_blocks", "off", "--vae",
                str(tmp_path / "vae.safetensors"), "--text_encoder", te_dir, "--sample_prompts",
                str(tmp_path / "prompts.txt"), "--sample_every_n_epochs", "1", "--sample_steps", "2",
                "--sample_width", "64", "--sample_height", "64", "--sample_cfg_scale", "3",
                "--family_option", "caption_dropout=0.5")
        log = r.stdout + r.stderr
        assert r.returncode == 0, log[-6000:]
        loras = sorted(f for f in os.listdir(out) if f.endswith(".safetensors"))
        sd = load_file(str(out / loras[-1]))
        assert any(k.startswith("lora_unet_layers_0_proj_q.") for k in sd)
        return
    # CPU: train.py runs on CUDA only, so the same steps in-process - the cached files through the driver, two
    # epochs of LoRA steps, a preview through the real text encoder and VAE, save and reload
    from fizgig.families.lora import FamilyLoRA
    drv = _drv()
    drv.set_options({"caption_dropout": "0.5"})
    files = sorted(os.listdir(cache))
    lats = [load_file(str(cache / f)) for f in files if f.endswith("_mossimage.safetensors")]
    texts = [load_file(str(cache / f)) for f in files if f.endswith("_mossimage_te.safetensors")]
    assert len(lats) == 3 and len(texts) == 3, files
    latents = [next(v for k, v in d.items() if k.startswith("latent_")) for d in lats]
    conds = [{"hidden_states": d["cond__hidden_states"][None]} for d in texts]
    assert latents[0].shape == (128, 4, 4) and conds[0]["hidden_states"].shape[-1] == 64
    dit = drv.load_dit(str(tmp_path / "dit.safetensors"), "cpu").float()
    drv.enable_gradient_checkpointing(dit, True)
    net = FamilyLoRA(dit, drv)
    net.add_trainable(4, 4.0)
    group = type("G", (), {"datasets": [type("D", (), {"cache_directory": str(cache)})()]})()
    drv.prepare_training(dit, group, net)
    assert drv._uncond is not None
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)
    losses = []
    for epoch in range(2):
        for z, c in zip(latents, conds):
            loss, info = drv.training_loss(dit, z[None], c, gen)
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
    assert all(math.isfinite(v) for v in losses)
    te = drv.load_text_encoder(te_dir, "cpu")
    pos, neg = drv.encode_text(te, ["a photo of sks", ""])
    drv.unload_text_encoder(te)
    # the empty prompt cached alongside the captions (same layer-major pass) is the one encoded on its own
    assert torch.equal(load_file(str(cache / "mossimage_empty_prompt.safetensors"))["hidden_states"],
                       neg["hidden_states"])
    vae = drv.load_vae(str(tmp_path / "vae.safetensors"), "cpu")
    with torch.no_grad():
        lat = drv.generate(dit, pos, 64, 64, steps=2, seed=1, cfg=3.0, neg_cond=neg)     # as train.py hands them
    img = drv.decode(vae, lat, 64, 64)
    assert img.size == (64, 64)
    img.save(str(tmp_path / "preview.png"))
    path = str(tmp_path / "tiny.safetensors")
    net.save(path)
    sd = load_file(path)
    assert any(k.startswith("lora_unet_layers_0_proj_q.") for k in sd) and all(torch.isfinite(v).all()
                                                                              for v in sd.values())
    net2 = FamilyLoRA(drv.load_dit(str(tmp_path / "dit.safetensors"), "cpu").float(), drv)
    net2.add_trainable(4, 4.0)
    assert net2.load_trainable(path) == 6 * TINY["num_layers"]
    RESULTS["T10 first/last loss"] = losses[0]
    RESULTS["T10 last loss"] = losses[-1]


def test_zz_print_results():
    for k, v in RESULTS.items():
        print(f"{k[:50]:50s} {v:.3e}")
