"""MOSS-Image DiT tests (T2, T3, T4): CPU, tiny random weights, parity against OpenMOSS's reference pipeline.

    python -m pytest tests -q -k moss
"""
import pytest
import torch

from moss_tiny import TINY, pair, rel

RESULTS = {}   # measured numbers, printed by -s


def _padded(rows, total=None):
    L = total or max(r.shape[0] for r in rows)
    embeds = torch.zeros(len(rows), L, rows[0].shape[-1])
    mask = torch.zeros(len(rows), L, dtype=torch.bool)
    for i, r in enumerate(rows):
        embeds[i, : r.shape[0]] = r
        embeds[i, r.shape[0]:] = 7.0          # junk in the padding: it must never matter
        mask[i, : r.shape[0]] = True
    return embeds, mask


# ---- T3: masked batch / padding (run first) ----------------------------------------------------------------------
@pytest.mark.parametrize("hw", [(3, 4), (4, 4), (1, 5)])
def test_t3_masked_batch_equals_reference_loop(hw):
    torch.manual_seed(1)
    fz, ref = pair()
    rows = [torch.randn(n, TINY["text_dim"]) for n in (3, 9, 6)]
    embeds, mask = _padded(rows)
    x = torch.randn(3, TINY["latent_channels"], *hw)
    t = torch.tensor([980.0, 431.0, 17.0])
    with torch.no_grad():
        want = ref(x, t, embeds, mask).sample
        loop = fz(x, t, embeds, mask, per_sample=True)
        masked = fz(x, t, embeds, mask)
        singles = torch.cat([fz(x[i:i + 1], t[i:i + 1], rows[i][None]) for i in range(3)])
    RESULTS[f"T3 loop {hw}"] = rel(loop, want)
    RESULTS[f"T3 masked {hw}"] = rel(masked, want)
    assert rel(loop, want) <= 1e-6
    assert rel(masked, want) <= 1e-5
    assert rel(singles, want) <= 1e-5


def test_t3_padding_one_prompt_changes_nothing():
    torch.manual_seed(2)
    fz, _ = pair()
    row = torch.randn(5, TINY["text_dim"])
    x = torch.randn(1, TINY["latent_channels"], 4, 3)
    embeds, mask = _padded([row], total=11)
    with torch.no_grad():
        a = fz(x, 300.0, row[None])
        b = fz(x, 300.0, embeds, mask)
        # masked-batch path forced: the same prompt twice, one copy padded
        e2, m2 = _padded([row, torch.cat([row, torch.randn(4, TINY["text_dim"])])])
        c = fz(torch.cat([x, x]), 300.0, e2, m2)[:1]
    RESULTS["T3 pad single"] = rel(b, a)
    RESULTS["T3 pad in batch"] = rel(c, a)
    assert rel(b, a) <= 1e-6 and rel(c, a) <= 1e-5


def test_t3_rejects_left_padding_and_empty_rows():
    fz, _ = pair()
    x = torch.randn(2, TINY["latent_channels"], 2, 2)
    e = torch.randn(2, 4, TINY["text_dim"])
    left = torch.tensor([[True] * 4, [False, True, True, True]])
    with pytest.raises(ValueError):
        fz(x, 1.0, e, left)
    with pytest.raises(ValueError):
        fz(x, 1.0, e, torch.tensor([[True] * 4, [False] * 4]))


def test_t3_masked_batch_bf16_report():
    torch.manual_seed(3)
    fz, ref = pair()
    fz, ref = fz.to(torch.bfloat16), ref.to(torch.bfloat16)
    rows = [torch.randn(n, TINY["text_dim"]) for n in (2, 8)]
    embeds, mask = _padded(rows)
    x = torch.randn(2, TINY["latent_channels"], 3, 3)
    with torch.no_grad():
        want = ref(x, torch.tensor([700.0, 50.0]), embeds, mask).sample.float()
        got = fz(x, torch.tensor([700.0, 50.0]), embeds, mask).float()
    RESULTS["T3 masked bf16"] = rel(got, want)
    assert rel(got, want) <= 2e-2


# ---- T4: forward parity --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("hw,skip", [((4, 4), ()), ((2, 6), (2,)), ((1, 7), (0, 3)), ((5, 3), ())])
def test_t4_forward_parity(hw, skip):
    torch.manual_seed(4)
    fz, ref = pair(seed=4)
    x = torch.randn(2, TINY["latent_channels"], *hw)
    c = torch.randn(2, 6, TINY["text_dim"])
    with torch.no_grad():
        want = ref(x, torch.tensor([999.0, 1.0]), c, skip_layers=list(skip)).sample
        got = fz(x, torch.tensor([999.0, 1.0]), c, skip_layers=skip)
    RESULTS[f"T4 {hw} skip{skip}"] = rel(got, want)
    assert rel(got, want) <= 1e-6


def test_t4_stages():
    from fizgig.moss_image import model as M
    fz, ref = pair(seed=5)
    hd = TINY["dim"] // TINY["num_heads"]
    want = ref.rope(3, 5, 7, torch.device("cpu"))
    got = M.rope_table(M.rope_bands(hd, 10000.0), 3, 5, 7, torch.device("cpu"))
    assert torch.equal(got, want)
    assert torch.equal(fz._bands, ref.rope.bands)
    x = torch.randn(1, 9, TINY["dim"])
    rope = want[:, :9]
    mod = torch.randn(1, 6, TINY["dim"])
    with torch.no_grad():
        assert torch.equal(fz.layers[1](x, rope, mod), ref.layers[1](x, rope, mod))
        t = torch.randn(1, 7, TINY["text_dim"])
        assert torch.equal(fz.text_embedder(t, want[:, 15:]), ref.text_embedder(t, want[:, 15:]))


def test_t4_gradient_checkpointing_same_grads():
    torch.manual_seed(6)
    fz, _ = pair(seed=6)
    x = torch.randn(1, TINY["latent_channels"], 3, 3)
    c = torch.randn(1, 4, TINY["text_dim"])
    grads = []
    for gc in (False, True):
        fz.zero_grad()
        fz.enable_gradient_checkpointing(gc)
        fz(x, 400.0, c).square().mean().backward()
        grads.append(fz.layers[2].proj["q"].weight.grad.clone())
    assert rel(grads[1], grads[0]) <= 1e-6


# ---- T2: loader --------------------------------------------------------------------------------------------------
def _full_sd(fz):
    return {k: v.clone() for k, v in fz.state_dict().items()}


def test_t2_fused_and_split_files_load_alike(tmp_path):
    import json

    from safetensors.torch import save_file

    from fizgig.moss_image.model import is_convrot, load_moss_dit
    fz, ref = pair(seed=7)
    sd = _full_sd(fz)
    fused = {}
    for k, v in sd.items():
        if k.endswith(".proj.q.weight"):
            stem = k[: -len("q.weight")]
            fused["diffusion_model." + stem + "qkv.weight"] = torch.cat(
                [sd[stem + "q.weight"], sd[stem + "k.weight"], sd[stem + "v.weight"]])
        elif not (k.endswith(".proj.k.weight") or k.endswith(".proj.v.weight")):
            fused["diffusion_model." + k] = v
    save_file(fused, str(tmp_path / "comfy.safetensors"))
    # diffusers layout: two shards + index, with the reference's rope.bands buffer
    tdir = tmp_path / "MOSS-Image-Base" / "transformer"
    tdir.mkdir(parents=True)
    keys = sorted(sd)
    half = {k: sd[k] for k in keys[: len(keys) // 2]}
    rest = {k: sd[k] for k in keys[len(keys) // 2:]}
    rest["rope.bands"] = ref.rope.bands.clone()
    save_file(half, str(tdir / "a.safetensors"))
    save_file(rest, str(tdir / "b.safetensors"))
    wm = {k: "a.safetensors" for k in half} | {k: "b.safetensors" for k in rest}
    (tdir / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    a = load_moss_dit(str(tmp_path / "comfy.safetensors"), device="cpu", dtype=torch.float32)
    b = load_moss_dit(str(tmp_path / "MOSS-Image-Base"), device="cpu", dtype=torch.float32)
    assert a.config == b.config == fz.config | {"rope_base": 10000.0}
    for k, v in sd.items():
        assert torch.equal(a.state_dict()[k], v) and torch.equal(b.state_dict()[k], v), k
    x, c = torch.randn(1, TINY["latent_channels"], 2, 3), torch.randn(1, 3, TINY["text_dim"])
    with torch.no_grad():
        assert torch.equal(a(x, 10.0, c), fz(x, 10.0, c))
    assert not is_convrot(str(tmp_path / "comfy.safetensors"))
    conv = dict(fused)
    conv["diffusion_model.layers.0.proj.qkv.comfy_quant"] = torch.zeros(4, dtype=torch.uint8)
    save_file(conv, str(tmp_path / "convrot.safetensors"))
    assert is_convrot(str(tmp_path / "convrot.safetensors"))
    with pytest.raises(ValueError, match="ConvRot"):
        load_moss_dit(str(tmp_path / "convrot.safetensors"), device="cpu")


def test_zz_print_results():
    for k, v in RESULTS.items():
        print(f"{k:32s} rel {v:.3e}")
