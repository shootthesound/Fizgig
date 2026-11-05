"""Batch size above 1 (Other Options): a batch of pictures whose captions differ in length gives each picture what the
one-picture forward gives it alone - tiny MOSS-Image and Qwen Image 2.1 models on the CPU, captions padded with a mask
- and a batch of one stacks as it always did."""
import torch

from fizgig.moss_image.model import MOSSImageDiT
from fizgig.qwen_image21.model import QwenImage21DiT


def _captions(lengths, dim, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, dim, generator=g) for n in lengths]


def _padded(hs):
    L = max(h.shape[0] for h in hs)
    return (torch.stack([torch.cat([h, h.new_zeros(L - h.shape[0], h.shape[1])]) for h in hs]),
            torch.stack([torch.arange(L) < h.shape[0] for h in hs]))


def test_moss_batch_matches_one_at_a_time():
    torch.manual_seed(0)
    dit = MOSSImageDiT(dim=64, ffn_dim=96, num_heads=4, num_layers=2, text_dim=24, freq_dim=32,
                       latent_channels=8).float().eval()
    caps = _captions((5, 9, 3), 24)
    x = torch.randn(3, 8, 4, 6)
    t = torch.tensor([150.0, 600.0, 930.0])
    with torch.no_grad():
        one = torch.cat([dit(x[i:i + 1], t[i:i + 1], caps[i][None]) for i in range(3)])
        enc, mask = _padded(caps)
        both = dit.forward_batch(x, t, enc, mask)
    assert torch.allclose(one, both, atol=1e-5, rtol=1e-4), (one - both).abs().max()


def test_qwen_batch_matches_one_at_a_time():
    torch.manual_seed(0)
    dit = QwenImage21DiT(num_layers=2, attention_head_dim=16, num_attention_heads=2, context_in_dim=24,
                         axes_dims_rope=(4, 6, 6)).float().eval()
    h = w = 4
    n = h * w
    caps = _captions((5, 9, 3), 24, seed=1)
    x = torch.randn(3, n, 64)
    t = torch.tensor([0.15, 0.6, 0.93])
    with torch.no_grad():
        one = []
        for i, c in enumerate(caps):
            L = c.shape[0]
            img_mask = torch.cat([torch.zeros(L, dtype=torch.bool), torch.ones(n // 4, dtype=torch.bool)])[None]
            one.append(dit(x[i:i + 1], c[None], t[i:i + 1], [[(1, h, w)]], img_mask,
                           torch.ones(1, L, dtype=torch.bool))[:, -n:])
        one = torch.cat(one)
        enc, mask = _padded(caps)
        L = enc.shape[1]
        img_mask = torch.cat([torch.zeros(L, dtype=torch.bool), torch.ones(n // 4, dtype=torch.bool)])[None]
        both = dit(x, enc, t, [[(1, h, w)]], img_mask, mask)[:, -n:]
    assert torch.allclose(one, both, atol=1e-5, rtol=1e-4), (one - both).abs().max()


def test_stack_conds_keeps_shapes_apart():
    from fizgig.families.driver import FamilyDriver
    a, b = {"x": torch.zeros(1, 4, 2)}, {"x": torch.zeros(1, 5, 2)}
    assert FamilyDriver.stack_conds(None, [a, b]) is None
    assert FamilyDriver.stack_conds(None, [a, dict(a)])["x"].shape == (2, 4, 2)


def test_batch_size_reaches_the_dataset_and_stands_down_what_reads_one_picture(tmp_path):
    import sys
    from PIL import Image
    from fizgig.families import form, launch
    from fizgig.families.registry import get as get_family
    moss = get_family("moss_image")
    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(2):
        Image.new("RGB", (64, 64)).save(d / f"p{i}.png")
        (d / f"p{i}.txt").write_text("zwxem a photo")
    models = {}
    for mf in moss.model_files:
        f = tmp_path / f"{mf.pref_key}.safetensors"
        f.write_bytes(b"0")
        models[mf.pref_key] = str(f)

    def plan(**values):
        v = {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"),
             "DATASET_CONFIG": str(tmp_path / "dataset.toml"), "KREA2_LOSS_WATCH": True, **values}
        inp = form.inputs(moss, v, models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder=str(d),
                          cache_root=str(tmp_path / "cache"), samples_dir=str(tmp_path / "out" / "sample"))
        return v, launch.plan(moss, inp)

    v, p = plan(BATCH_SIZE="3")
    train = " ".join(next(s.cmd for s in p.stages if "train.py" in " ".join(s.cmd)))
    assert "batch_size = 3" in p.dataset_toml and "--log_per_image_loss" not in train
    assert "KREA2_LOSS_WATCH" in form.state(moss, v, str(d))["disabled"]
    v, p = plan(BATCH_SIZE="1")
    train = " ".join(next(s.cmd for s in p.stages if "train.py" in " ".join(s.cmd)))
    assert "batch_size = 1" in p.dataset_toml and "--log_per_image_loss" in train
    v = {**form.DEFAULTS, "BATCH_SIZE": "3", "FAMILY_KIND": "slider", "FAMILY_SLIDER": True}   # one pair a step
    assert launch.batch_size(moss, v) == 1 and "BATCH_SIZE" in form.state(moss, v, str(d))["disabled"]


def test_batch_plan_note_asks_what_start_sends(tmp_path):
    """The Batch size row's plan note asks the planner with the precision, swap and size Start hands the trainer, and
    says when the batch would swap where batch 1 doesn't."""
    import re
    import sys
    from PIL import Image
    from fizgig.families import form, launch
    from fizgig.families.registry import get as get_family
    from fizgig.web import batch_plan
    d = tmp_path / "imgs"
    d.mkdir()
    Image.new("RGB", (64, 64)).save(d / "p0.png")
    (d / "p0.txt").write_text("x")
    for fam in ("qwen_image21", "moss_image", "klein"):
        desc = get_family(fam)
        models = {}
        for mf in desc.model_files:
            f = tmp_path / f"{fam}_{mf.pref_key}.safetensors"
            f.write_bytes(b"0")
            models[mf.pref_key] = str(f)
        labels = {**launch.PRECISION_LABELS, **desc.precision_labels}
        for prec, swap, mp in (("auto", "Auto (detect from GPU)", "0.25"), ("bf16", "Auto (detect from GPU)", "1.0"),
                               ("auto", "8", "0.5"), ("int8", "0", "0.25")):
            v = {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"),
                 "DATASET_CONFIG": str(tmp_path / "dataset.toml"), "BATCH_SIZE": "2",
                 "FAMILY_PRECISION": labels[prec], "BLOCKS_SWAP": swap, "DATASET_MEGAPIXELS": mp}
            inp = form.inputs(desc, v, models=models, python=sys.executable, repo_dir=str(tmp_path),
                              image_folder=str(d), cache_root=str(tmp_path / "cache"),
                              samples_dir=str(tmp_path / "out" / "sample"))
            p = launch.plan(desc, inp)
            train = next(s.cmd for s in p.stages if "train.py" in " ".join(s.cmd))
            sent_prec = train[train.index("--precision") + 1] if "--precision" in train else "bf16"
            sent_swap = int(train[train.index("--blocks_to_swap") + 1]) if "--blocks_to_swap" in train else 0
            side = int(re.search(r"resolution = \[(\d+), \d+\]", p.dataset_toml).group(1))
            req = batch_plan.request(desc, v)
            if sent_prec != "auto" and sent_swap >= 0:
                assert req is None, (fam, prec, swap)
                continue
            assert req == (sent_prec, sent_swap, side * side / 1e6, 2), (fam, prec, swap, mp, req)
    assert batch_plan.request(get_family("moss_image"), {**form.DEFAULTS, "BATCH_SIZE": "1"}) is None
    n = batch_plan.note(None, ("bf16", 0, 31.0), ("bf16", 6, 31.0), 2)
    assert "6 blocks are swapped (batch 1: none)" in n and "Batch 1 is likely faster" in n
    assert "still be quicker" in batch_plan.note(None, ("bf16", 8, 15.0), ("bf16", 16, 15.0), 3)
    assert "About as quick" in batch_plan.note(None, ("bf16", 12, 15.0), ("bf16", 25, 15.0), 2)
    assert "fits as batch 1 does" in batch_plan.note(None, ("int8", 0, 23.0), ("int8", 0, 23.0), 3)
    assert "drops to 4-bit NF4 (batch 1: int8)" in batch_plan.note(None, ("int8", 0, 15.0), ("nf4", 0, 15.0), 2)


def _driver_parity(drv, dit, lat, conds, mask, autocast=None):
    """The batched loss against the mean of the pictures' own, from the same generator."""
    import contextlib
    ctx = torch.autocast("cpu", dtype=autocast) if autocast else contextlib.nullcontext()
    with torch.no_grad(), ctx:
        b, _ = drv.training_loss_batch(dit, lat, drv.stack_conds(conds), torch.Generator().manual_seed(7),
                                       loss_mask=mask)
        g = torch.Generator().manual_seed(7)
        s = sum(drv.training_loss(dit, lat[i:i + 1], conds[i], g,
                                  loss_mask=None if mask is None else mask[i:i + 1])[0] for i in range(len(conds)))
    return float(b), float(s) / len(conds)


def test_fibo_batch_matches_one_at_a_time():
    from fizgig.families.registry import get as get_family
    from fizgig.fibo2.model import Fibo2DiT
    torch.manual_seed(0)
    dit = Fibo2DiT(dim=64, n_layers=5, n_refiner_layers=1, n_heads=4, cap_feat_dim=24,
                   injection_layer_ids=(0, 1, 2, 3, 4), perceiver_num_layers=1, min_num_gist_tokens=4,
                   max_num_gist_tokens=8, gist_step=2, gist_min_text_len=2, gist_max_text_len=12, axes_dims=(4, 6, 6),
                   axes_lens=(64, 32, 32)).float().eval()
    drv = get_family("fibo2").load_driver()
    g = torch.Generator().manual_seed(1)
    conds = [{"tokens": torch.randn(1, 1, 6, n, 24, generator=g), "mask": torch.ones(1, 1, n, dtype=torch.bool)}
             for n in (5, 11, 3)]                       # three gist counts: the batch pads the gist tokens too
    lat = torch.randn(3, 32, 8, 8, generator=g)
    for mask in (None, torch.rand(3, 1, 8, 8, generator=g)):
        b, s = _driver_parity(drv, dit, lat, conds, mask)
        assert abs(b - s) <= 1e-5 * abs(s), (mask is None, b, s)


def test_flux3_batch_matches_one_at_a_time(monkeypatch):
    from fizgig.families.registry import get as get_family
    from fizgig.flux3 import model as M
    for k, v in (("HIDDEN", 64), ("HEADS", 4), ("DEPTH", 1), ("DEPTH_SINGLE", 2), ("AXES_DIM", (4, 4, 4, 4))):
        monkeypatch.setattr(M, k, v)
    monkeypatch.setitem(M.IN_CHANNELS, "text", 24)
    torch.manual_seed(0)
    dit = M.Flux3DiT().float().eval()
    drv = get_family("flux3").load_driver()
    g = torch.Generator().manual_seed(1)
    conds = [{"tokens": torch.randn(1, n, 24, generator=g)} for n in (5, 11, 3)]
    lat = torch.randn(3, 128, 4, 6, generator=g)
    for mask in (None, torch.rand(3, 1, 4, 6, generator=g)):
        b, s = _driver_parity(drv, dit, lat, conds, mask, autocast=torch.bfloat16)
        assert abs(b - s) <= 1e-5 * abs(s), (mask is None, b, s)


def test_krea2_batch_matches_one_at_a_time(monkeypatch):
    from fizgig.families.registry import get as get_family
    from fizgig.krea2 import driver as KD
    from fizgig.krea2.model import SingleMMDiTConfig, SingleStreamDiT
    monkeypatch.setattr(KD, "DTYPE", torch.float32)
    torch.manual_seed(0)
    dit = SingleStreamDiT(SingleMMDiTConfig(features=64, tdim=32, txtdim=24, heads=4, multiplier=2, layers=2, patch=2,
                                            channels=16, txtlayers=3, txtheads=4, txtkvheads=4)).float().eval()
    drv = get_family("krea2").load_driver()
    g = torch.Generator().manual_seed(1)
    conds = []
    for n in (5, 11, 3):                                # the encoder's layout: prompt, padding, template suffix
        m = torch.zeros(1, 20, dtype=torch.bool)
        m[0, :n] = True
        m[0, -2:] = True
        conds.append({"hidden_states": torch.randn(1, 20, 3, 24, generator=g), "attention_mask": m})
    lat = torch.randn(3, 16, 8, 8, generator=g)
    for mask in (None, torch.rand(3, 1, 8, 8, generator=g)):
        b, s = _driver_parity(drv, dit, lat, conds, mask)
        assert abs(b - s) <= 1e-5 * abs(s), (mask is None, b, s)
