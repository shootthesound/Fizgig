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
