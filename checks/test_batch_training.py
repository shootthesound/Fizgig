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
