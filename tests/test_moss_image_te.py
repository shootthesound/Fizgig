"""MOSS-Image text path (T9): the side venv, te_worker through the real proxy protocol, against an in-process
Qwen3_5Model (tiny random weights) - from Fizgig's own venv, whose transformers stays 4.57.

Needs the side venv (envs/moss_te, made on first run with network access; FIZGIG_MOSS_TE_ENV points elsewhere) and the MOSS
tokenizer (FIZGIG_MOSS_TOKENIZER, the cached helper repo, or the Hub); skipped when either is unavailable.
"""
import json
import os
import subprocess

import pytest
import torch

HERE = os.path.dirname(__file__)
CAPTIONS = ["A cat", "a red fox in the snow, {braces}\nline two, ünïcode 漢字", "", "word " * 300]


def _tokenizer():
    cand = os.environ.get("FIZGIG_MOSS_TOKENIZER", "/tmp/moss/tokenizer")      # (the box's copy by default)
    if os.path.isfile(os.path.join(cand, "tokenizer.json")):
        return cand
    from fizgig.moss_image.te_proxy import resolve_tokenizer
    t = resolve_tokenizer("/nonexistent/te")
    if os.path.isdir(t):
        return t
    try:
        from huggingface_hub import snapshot_download
        d = snapshot_download("OpenMOSS-Team/MOSS-Image-Base", allow_patterns=["tokenizer/*"])
        return os.path.join(d, "tokenizer")
    except Exception as e:                                # noqa: BLE001
        pytest.skip(f"MOSS tokenizer unavailable: {e}")


@pytest.fixture(scope="module")
def side(tmp_path_factory):
    from fizgig.moss_image import te_env
    try:
        py = te_env.ensure()
    except Exception as e:                                # noqa: BLE001
        pytest.skip(f"side venv unavailable: {e}")
    tok = _tokenizer()
    out = tmp_path_factory.mktemp("moss_te")
    caps = out / "captions.json"
    caps.write_text(json.dumps(CAPTIONS), encoding="utf-8")
    r = subprocess.run([py, os.path.join(HERE, "moss_te_tiny_build.py"), str(out), tok, str(caps)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return py, tok, out


def test_t9_main_venv_untouched(side):
    import transformers
    assert transformers.__version__ == "4.57.6"
    from fizgig.moss_image import te_env
    st = te_env.status(side[0])
    assert st["ok"] and int(st["transformers"].split(".")[0]) >= 5
    assert st["torch"] == torch.__version__          # the side venv runs Fizgig's own torch


def test_t9_template_string(side):
    from fizgig.moss_image.te_worker import TEMPLATE
    texts = json.loads((side[2] / "texts.json").read_text())
    for c, t in zip(CAPTIONS, texts):
        assert t == TEMPLATE.format(c.strip())      # the jinja trims the message (ComfyUI's plugin doesn't)


def test_t9_worker_equals_in_process_model(side):
    from safetensors.torch import load_file

    from fizgig.moss_image.te_proxy import MossTextEncoder
    py, tok, out = side
    ref = load_file(str(out / "reference.safetensors"))
    te = MossTextEncoder(str(out / "text_encoder"), device="cpu", dtype="float32", out_dtype="float32",
                         python=py, tokenizer=tok)
    try:
        got = te.encode(CAPTIONS)
        again = te.encode(CAPTIONS[:1])                  # the worker serves several requests
    finally:
        te.unload()
    assert te._proc is None
    worst = 0.0
    for i, g in enumerate(got):
        r = ref[f"h{i}"]
        assert g.shape == r.shape and g.shape[-1] == 64
        worst = max(worst, float((g - r).abs().max()))
    print(f"T9 worker vs in-process Qwen3_5Model max abs {worst:.3e}")
    assert worst == 0.0
    assert torch.equal(again[0], got[0])
    assert got[2].shape[0] >= 8                     # the empty prompt is still a templated prompt


def test_t9_truncation_and_bf16(side):
    from fizgig.moss_image.te_proxy import MossTextEncoder
    py, tok, out = side
    te = MossTextEncoder(str(out / "text_encoder"), device="cpu", python=py, tokenizer=tok, max_tokens=50)
    try:
        h = te.encode(["word " * 300, "A cat"])
    finally:
        te.unload()
    assert h[0].shape[0] == 50 and h[0].dtype == torch.bfloat16
    assert h[1].shape[0] < 50


def test_t9_crash_gives_a_clear_error(side, tmp_path):
    from fizgig.moss_image.te_proxy import MossTextEncoder
    py, tok, out = side
    with pytest.raises(RuntimeError, match="text encoder"):
        MossTextEncoder(str(tmp_path / "missing"), device="cpu", python=py, tokenizer=tok)


def test_t9_offline(side, monkeypatch):
    """Local tokenizer + weights: the worker runs with HF_HUB_OFFLINE=1 (set by the proxy)."""
    from fizgig.moss_image.te_proxy import MossTextEncoder
    py, tok, out = side
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")     # any network use would fail
    te = MossTextEncoder(str(out / "text_encoder"), device="cpu", python=py, tokenizer=tok)
    try:
        assert te.encode(["x"])[0].shape[0] > 0
    finally:
        te.unload()


# ---- layer-major stream mode (the text cache's fast path) ---------------------------------------------------------
@pytest.mark.parametrize("weights,order,group_tokens", [("resident", "layer", None), ("resident", "caption", None),
                                                        ("disk", "layer", None), ("disk", "caption", None),
                                                        ("resident", "layer", 40), ("disk", "layer", 40)])
def test_t9_stream_orders_equal_reference(side, weights, order, group_tokens):
    """Every stream variant through the real worker, fp32: bit-for-bit the in-process model, whatever the caption
    lengths (12 ... 300+ tokens) and however the captions are grouped (40 tokens: one caption per group)."""
    from safetensors.torch import load_file

    from fizgig.moss_image.te_proxy import MossTextEncoder
    py, tok, out = side
    ref = load_file(str(out / "reference.safetensors"))
    te = MossTextEncoder(str(out / "text_encoder"), device="cpu", mode="stream", dtype="float32", out_dtype="float32",
                         python=py, tokenizer=tok, weights=weights, order=order, group_tokens=group_tokens)
    try:
        assert te.info["weights"] == weights and te.info["order"] == order
        got = te.encode(CAPTIONS)
        again = te.encode(CAPTIONS[1:2])                 # weights go home after a pass and come back for the next
    finally:
        te.unload()
    worst = max(float((g - ref[f"h{i}"]).abs().max()) for i, g in enumerate(got))
    print(f"T9 stream {weights}/{order}/{group_tokens} vs in-process max abs {worst:.3e}")
    assert [g.shape for g in got] == [ref[f"h{i}"].shape for i in range(len(CAPTIONS))]
    assert worst == 0.0 and torch.equal(again[0], got[1])


def test_t9_layer_major_bf16_equals_per_caption(side):
    """bf16 (the real setting): the layer-major pass returns exactly what the old per-caption stream returned."""
    from fizgig.moss_image.te_proxy import MossTextEncoder
    py, tok, out = side
    res = {}
    for order in ("caption", "layer"):
        te = MossTextEncoder(str(out / "text_encoder"), device="cpu", mode="stream", python=py, tokenizer=tok,
                             weights="resident", order=order)
        try:
            res[order] = te.encode(CAPTIONS)
        finally:
            te.unload()
    for a, b in zip(res["caption"], res["layer"]):
        assert a.dtype == torch.bfloat16 and torch.equal(a, b)


def test_pick_weights_by_free_ram(monkeypatch, tmp_path):
    from fizgig.moss_image import te_proxy as P
    (tmp_path / "model.safetensors").write_bytes(b"\0" * (1 << 20))
    monkeypatch.delenv("FIZGIG_MOSS_TE_WEIGHTS", raising=False)
    monkeypatch.setattr(P, "weights_gb", lambda p: 17.9)          # the 9B's text model + vision shards
    monkeypatch.setattr(P, "available_ram_gb", lambda: 50.0)
    assert P.pick_weights(str(tmp_path)) == "resident"          # 64 GB machine
    monkeypatch.setattr(P, "available_ram_gb", lambda: 20.0)
    assert P.pick_weights(str(tmp_path)) == "disk"              # 32 GB machine with a browser open
    monkeypatch.setattr(P, "available_ram_gb", lambda: None)
    assert P.pick_weights(str(tmp_path)) == "resident"          # no psutil: the old behaviour
    monkeypatch.setenv("FIZGIG_MOSS_TE_WEIGHTS", "disk")
    monkeypatch.setattr(P, "available_ram_gb", lambda: 50.0)
    assert P.pick_weights(str(tmp_path)) == "disk"
    monkeypatch.setenv("FIZGIG_MOSS_TE_ORDER", "caption")
    assert P.pick_order() == "caption"
    monkeypatch.delenv("FIZGIG_MOSS_TE_ORDER")
    assert P.pick_order() == "layer"
    monkeypatch.undo()
    assert abs(P.weights_gb(str(tmp_path)) - 1 / 1024) < 1e-9
