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
    assert "batch_size = 3" in p.dataset_toml and "--log_per_image_loss" in train     # weighed picture by picture
    assert "KREA2_LOSS_WATCH" not in form.state(moss, v, str(d))["disabled"]
    v, p = plan(BATCH_SIZE="1")
    train = " ".join(next(s.cmd for s in p.stages if "train.py" in " ".join(s.cmd)))
    assert "batch_size = 1" in p.dataset_toml and "--log_per_image_loss" in train
    v, p = plan(BATCH_SIZE="3", FAMILY_KIND="finetune", FAMILY_FT=True)          # a fine-tune batches too
    assert "batch_size = 3" in p.dataset_toml and launch.batch_size(moss, v) == 3
    assert "BATCH_SIZE" not in form.state(moss, v, str(d))["disabled"] and form.batch_blockers(moss, v) == []
    from fizgig.web import batch_plan
    assert batch_plan.request(moss, v) is None          # its windows are planned by the fine-tune card's line
    v = {**form.DEFAULTS, "BATCH_SIZE": "3", "FAMILY_KIND": "slider", "FAMILY_SLIDER": True}   # one pair a step
    assert launch.batch_size(moss, v) == 1 and "BATCH_SIZE" in form.state(moss, v, str(d))["disabled"]


def test_auto_plans_each_extra_picture_from_its_measured_cost():
    """A batch's plan adds batch_gb's (fixed + per megapixel) for each picture past the first - train_memory's own
    curve can be flat across sizes (its peak is the preview's), so stretching it by the batch would plan nothing more."""
    from types import SimpleNamespace
    from fizgig.families import quant
    desc = SimpleNamespace(train_memory={"bf16": (((0.25, 10.0), (1.0, 10.0)), 0.5)}, precisions=("bf16",),
                           auto_precisions=(), batch_gb=(1.0, 2.0), display_name="t")
    driver = SimpleNamespace(max_blocks_to_swap=lambda: 20)
    assert quant.plan(desc, driver, "bf16", -1, free_gb=12.0, megapixels=1.0)[:2] == ("bf16", 0)
    assert quant.plan(desc, driver, "bf16", -1, free_gb=12.0, megapixels=1.0, pictures=3)[:2] == ("bf16", 11)


def test_every_batching_family_has_its_extra_picture_cost():
    from fizgig.families.registry import training_families
    missing = [d.key for d in training_families() if d.batch_training and not any(d.batch_gb)]
    assert not missing, missing


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
    """The batched loss against the mean of the pictures' own, from the same generator - and each picture's loss in
    the batch ("per", what the loss watch weighs and records) against its own step's, its t against its own."""
    import contextlib
    ctx = torch.autocast("cpu", dtype=autocast) if autocast else contextlib.nullcontext()
    with torch.no_grad(), ctx:
        b, info = drv.training_loss_batch(dit, lat, drv.stack_conds(conds), torch.Generator().manual_seed(7),
                                          loss_mask=mask)
        g = torch.Generator().manual_seed(7)
        one = [drv.training_loss(dit, lat[i:i + 1], conds[i], g, loss_mask=None if mask is None else mask[i:i + 1])
               for i in range(len(conds))]
    s = sum(float(l) for l, _ in one)
    per = [float(x) for x in info["per"]]
    assert len(per) == len(one) and abs(sum(per) / len(per) - float(b)) <= 1e-6 * abs(float(b))
    for p, (l, i1), t in zip(per, one, info["ts"]):
        assert abs(p - float(l)) <= 5e-3 * abs(float(l)), (p, float(l))
        assert abs(t - float(i1.get("t", t))) <= 1e-6, (t, i1.get("t"))
    return float(b), s / len(conds)


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


def test_h3_batch_matches_one_at_a_time(monkeypatch):
    """H3 packs the pictures one after another (attention over each picture's own rows) - the silence rows its forward
    draws come from torch's own stream, so both sides start that stream from the same seed."""
    from fizgig.families.registry import get as get_family
    from fizgig.minimax import driver as HD
    from fizgig.minimax.model import MiniMaxH3Config, MiniMaxH3DiT
    monkeypatch.setattr(HD, "DTYPE", torch.float32)
    torch.manual_seed(0)
    dit = MiniMaxH3DiT(MiniMaxH3Config(hidden_size=128, num_layers=2, token_refiner_num_layers=1, num_attention_heads=1,
                                       attention_head_dim=128, ffn_hidden_size=128, text_dim=24,
                                       time_embed_hidden_size=64, time_embed_dim=32)).float().eval()
    drv = get_family("minimax").load_driver()
    g = torch.Generator().manual_seed(1)
    conds = [{"hidden_states": torch.randn(1, n, 24, generator=g)} for n in (5, 11, 3)]
    lat = torch.randn(3, 24, 8, 6, generator=g)
    for mask in (None, torch.rand(3, 1, 8, 6, generator=g)):
        torch.manual_seed(5)
        with torch.no_grad():
            b, info = drv.training_loss_batch(dit, lat, drv.stack_conds(conds), torch.Generator().manual_seed(7),
                                              loss_mask=mask)
        torch.manual_seed(5)
        g7 = torch.Generator().manual_seed(7)
        with torch.no_grad():
            one = [float(drv.training_loss(dit, lat[i:i + 1], conds[i], g7,
                                           loss_mask=None if mask is None else mask[i:i + 1])[0]) for i in range(3)]
        s = sum(one) / 3
        assert abs(float(b) - s) <= 1e-5 * abs(s), (mask is None, float(b), s)
        for p, o in zip(info["per_raw"], one):           # each picture's own loss, as the loss watch records it
            assert abs(p - o) <= 1e-5 * abs(o), (p, o)


def test_h3_reference_distillation_trains_one_picture_a_step():
    from fizgig.families import form, launch
    from fizgig.families.registry import get as get_family
    h3 = get_family("minimax")
    on = {**form.DEFAULTS, "H3_DISTILL": True}
    assert "Learn identity from my dataset (reference distillation)" in form.batch_blockers(h3, on)
    st = launch.option_tokens(h3, {"BATCH_SIZE": "2", "FAMILY_OPTIONS": {"H3_DISTILL": True}})[0]
    assert not any("distill" in str(a) for a in st), st


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


def test_auto_plans_each_extra_picture_from_its_measured_cost():
    """A batch's plan adds batch_gb's (fixed + per megapixel) for each picture past the first - train_memory's own curve
    can be flat across sizes (its peak is the preview's), so stretching it by the batch would plan nothing more."""
    from types import SimpleNamespace
    from fizgig.families import quant
    desc = SimpleNamespace(train_memory={"bf16": (((0.25, 10.0), (1.0, 10.0)), 0.5)}, precisions=("bf16",),
                           auto_precisions=(), batch_gb=(1.0, 2.0), display_name="t")
    driver = SimpleNamespace(max_blocks_to_swap=lambda: 20)
    assert quant.plan(desc, driver, "bf16", -1, free_gb=12.0, megapixels=1.0)[:2] == ("bf16", 0)
    p, n, _ = quant.plan(desc, driver, "bf16", -1, free_gb=12.0, megapixels=1.0, pictures=3)   # 16 GB needed
    assert (p, n) == ("bf16", 11) and quant.fits(desc, "bf16", 11, 12.0, 1.0, 3)
    assert not quant.fits(desc, "bf16", 0, 12.0, 1.0, 3) and quant.fits(desc, "bf16", 0, 12.0, 1.0, 1)


def test_batch_blockers_name_what_trains_one_picture_a_step():
    from fizgig.families import form
    from fizgig.families.registry import get as get_family
    moss = get_family("moss_image")
    assert form.batch_blockers(moss, dict(form.DEFAULTS)) == []
    assert form.batch_blockers(moss, {**form.DEFAULTS, "KREA2_LOSS_WATCH": True}) == []     # weighed per picture
    assert form.batch_blockers(moss, {**form.DEFAULTS, "STAGE_MP": "0.25"}) == ["Start at a smaller size"]
    assert form.batch_blockers(get_family("acestep15xl"), dict(form.DEFAULTS)) == ["this model trains one picture a step"]


def test_h3_batch_note_and_move_ask_its_own_planner(tmp_path, monkeypatch):
    """H3 has no train_memory figures: the Batch size note and a moved run's batch ask its driver's planner (the one
    Start runs) with the run's DiT file, and a bigger batch never plans less memory."""
    import json
    import struct
    from fizgig.families import form
    from fizgig.families.registry import get as get_family
    from fizgig.utils import device
    from fizgig.web import batch_plan
    h3 = get_family("minimax")
    hdr = json.dumps({"adaln_t_table": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    dit = tmp_path / "h3.safetensors"
    dit.write_bytes(struct.pack("<Q", len(hdr)) + hdr + b"\0" * 4)
    v = {**form.DEFAULTS, "BATCH_SIZE": "4", "DATASET_MEGAPIXELS": "1.0", "BLOCKS_SWAP": "Auto (detect from GPU)"}
    assert batch_plan.request(h3, v) is not None
    assert batch_plan.run_for(h3, v, {}) is None                       # no DiT file here: no note, a move stays at 1
    run = batch_plan.run_for(h3, v, batch_plan.models(h3, {"minimax_dit": str(dit)}.get))
    assert run and run["dit_path"] == str(dit)
    drv = h3.load_driver()
    for free in (94.0, 30.0, 18.0):
        monkeypatch.setattr(device, "plannable_free_vram", lambda free=free: free)
        plans = [drv.plan_pictures("auto", -1, megapixels=1.0, pictures=n, run=run) for n in (1, 2, 4, 8)]
        assert all(p in ("int8", "nf4") for p, _ in plans), plans
        for (p1, s1), (p2, s2) in zip(plans, plans[1:]):            # more pictures: more swap, or a smaller base
            assert (p2 == p1 and s2 >= s1) or (p1, p2) == ("int8", "nf4"), (free, plans)
    monkeypatch.setattr(device, "plannable_free_vram", lambda: 94.0)
    assert drv.plan_pictures("auto", -1, megapixels=1.0, pictures=8, run=run) == ("int8", 0)


def test_picture_losses_average_to_the_batch_loss():
    """Each picture's loss (what the loss watch weighs and records at Batch size above 1) averages to the batch's
    loss, with and without per-cell weights."""
    import torch.nn.functional as F
    from fizgig.families.driver import picture_losses, weighted_loss
    g = torch.Generator().manual_seed(3)
    se, mw = torch.rand(4, 7, 9, generator=g), torch.rand(4, 7, 9, generator=g)
    assert torch.allclose(picture_losses(se).mean(), se.mean())
    assert torch.allclose(picture_losses(se, mw).mean(), weighted_loss(se, mw))
    assert torch.allclose(picture_losses(se, mw)[1], weighted_loss(se[1:2], mw[1:2]))
    a, b = torch.rand(3, 5, generator=g), torch.rand(3, 5, generator=g)
    assert torch.allclose(picture_losses((a - b).pow(2)), torch.stack([F.mse_loss(a[i], b[i]) for i in range(3)]))


def test_a_batch_weighs_and_records_each_picture_as_its_own_step(tmp_path):
    """Batch size above 1 with the loss watch: each picture's loss at its own per-image LR, an excluded one at 0 and
    out of the average and the records, a regularisation picture at its fixed multiplier, one record per picture."""
    from fizgig.families.loss_watch import Watch

    class Fake:
        def __init__(self):
            self.seen = []

        def is_excluded(self, keys):
            return keys == ["c"]

        def multiplier(self, keys):
            return {"a": 0.5, "b": 1.0}.get(keys[0], 1.0)

        def observe(self, **kw):
            self.seen.append((kw["item_keys"], kw["timestep"], kw["loss"]))

    w = Watch.__new__(Watch)
    w.watch = Fake()
    per = torch.tensor([0.2, 0.4, 0.6, 0.8], requires_grad=True)
    info = {"per": per, "ts": [0.1, 0.2, 0.3, 0.4]}
    scaled, rec, kept = w.weigh(info, ["a", "b", "c", "r"], reg_keys={"r"}, reg_mult=2.0)
    assert kept == [0, 1, 3]
    assert abs(float(scaled) - (0.2 * 0.5 + 0.4 + 0.8 * 2.0) / 3) < 1e-6
    assert abs(float(rec) - (0.2 + 0.4 + 0.8) / 3) < 1e-6        # the raw losses, as batch 1 records them
    scaled.backward()
    assert per.grad.tolist()[2] == 0.0                            # the excluded picture trains nothing
    w.observe_each(1, 5, ["a", "b", "c", "r"], info["ts"], [0.2, 0.4, 0.6, 0.8], kept)
    assert w.watch.seen == [(["a"], 0.1, 0.2), (["b"], 0.2, 0.4), (["r"], 0.4, 0.8)]
    assert w.excluded({"item_keys": ["c", "c"]}) and not w.excluded({"item_keys": ["a", "c"]})
    w.watch = None                                                # the watch off: the plain mean
    scaled, rec, kept = w.weigh(info, ["a", "b", "c", "r"])
    assert abs(float(scaled) - 0.5) < 1e-6 and kept == [0, 1, 2, 3]


def test_moss_batch_runs_each_picture_alone():
    """#221: MOSS's batch runs each picture through the DiT alone at its own caption length (as OneTrainer does) - no
    padding, no mask - so the batched loss is the pictures' own losses, mask or not."""
    from fizgig.families.registry import get as get_family
    torch.manual_seed(0)
    dit = MOSSImageDiT(dim=64, ffn_dim=96, num_heads=4, num_layers=2, text_dim=24, freq_dim=32,
                       latent_channels=8).float().eval()
    drv = get_family("moss_image").load_driver()
    g = torch.Generator().manual_seed(2)
    conds = [{"hidden_states": torch.randn(1, n, 24, generator=g)} for n in (5, 9, 3)]
    assert "items" in drv.stack_conds(conds)
    lat = torch.randn(3, 8, 4, 6, generator=g)
    for mask in (None, torch.rand(3, 1, 4, 6, generator=g)):
        b, s = _driver_parity(drv, dit, lat, conds, mask)
        assert abs(b - s) <= 1e-5 * abs(s), (mask is None, b, s)

def test_qwen_padding_to_16_matches_one_at_a_time():
    """#221: Qwen Image 2.1's batches pad captions to a multiple of 16 too; the extra keys are masked, so each picture
    still gets what it gets alone (on the GPU too, through the cuDNN path)."""
    from fizgig.qwen_image21.driver import QwenImage21Driver
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    dit = QwenImage21DiT(num_layers=2, attention_head_dim=16, num_attention_heads=2, context_in_dim=24,
                         axes_dims_rope=(4, 6, 6)).float().eval().to(dev)
    h = w = 4
    n = h * w
    caps = [c.to(dev) for c in _captions((5, 9, 3), 24, seed=3)]
    x = torch.randn(3, n, 64, device=dev)
    t = torch.tensor([0.15, 0.6, 0.93], device=dev)
    with torch.no_grad():
        one = []
        for i, c in enumerate(caps):
            L = c.shape[0]
            img_mask = torch.cat([torch.zeros(L, dtype=torch.bool), torch.ones(n // 4, dtype=torch.bool)])[None].to(dev)
            one.append(dit(x[i:i + 1], c[None], t[i:i + 1], [[(1, h, w)]], img_mask,
                           torch.ones(1, L, dtype=torch.bool, device=dev))[:, -n:])
        one = torch.cat(one)
        cond = QwenImage21Driver.stack_conds(None, [{"hidden_states": c[None]} for c in caps])
        enc, mask = cond["hidden_states"], cond["mask"]
        assert enc.shape[1] == 16
        img_mask = torch.cat([torch.zeros(16, dtype=torch.bool), torch.ones(n // 4, dtype=torch.bool)])[None].to(dev)
        both = dit(x, enc, t, [[(1, h, w)]], img_mask, mask)[:, -n:]
    assert torch.allclose(one, both, atol=1e-4, rtol=1e-3), (one - both).abs().max()
