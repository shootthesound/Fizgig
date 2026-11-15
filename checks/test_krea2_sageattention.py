"""Inference backend selection without GPU kernels or model downloads."""
import json
import runpy
import shutil
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from fizgig.krea2 import attention as A
from fizgig.krea2 import generation_attention as G


@pytest.mark.parametrize("backend,installed,want", [("rocm", False, True), ("cuda", False, False),
                                                    ("", False, False), ("cuda", True, True)])
def test_preference_offered_only_on_amd_or_with_installed_package(monkeypatch, backend, installed, want):
    monkeypatch.setenv("FIZGIG_GPU_BACKEND", backend)
    monkeypatch.setattr(G.importlib.util, "find_spec", lambda name: object() if installed else None)
    assert G.preference_available() is want


def test_preference_ui_hides_sage_controls_without_capability():
    if not shutil.which("node"):
        pytest.skip("node not installed")
    app = Path(__file__).resolve().parents[1] / "src/fizgig/web/static/app.js"
    source = app.read_text()
    card = source[source.index("const INFERENCE_SWAP"):source.index("// where the LoRA and reference pickers")]
    script = '''
const assert = require('node:assert/strict');
const S = {set: {}, sageAttentionOption: false};
function el(tag, props, ...children) {
  return {tag, props, children, append(...items) {this.children.push(...items);}, addEventListener() {}};
}
''' + card + '''
assert(!JSON.stringify(inferenceCard()).includes('Force SageAttention'));
S.sageAttentionOption = true;
assert(JSON.stringify(inferenceCard()).includes('Force SageAttention'));
'''
    subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)


def _prefs(monkeypatch, tmp_path, on):
    path = tmp_path / "prefs.json"
    path.write_text(json.dumps({"inference_sageattention": on}))
    monkeypatch.setattr(G.user_data, "PREFS_FILE", str(path))
    return path


def _sage(q, k, v, *, tensor_layout, is_causal):
    assert tensor_layout == "HND" and not is_causal
    return F.scaled_dot_product_attention(q, k, v, enable_gqa=q.shape[1] != k.shape[1])


def test_rocm_package_does_not_need_varlen(monkeypatch):
    monkeypatch.setitem(sys.modules, "sageattention", SimpleNamespace(sageattn=_sage))
    module = runpy.run_path(A.__file__)
    assert module["sageattn"] is _sage
    assert module["sageattn_varlen"] is None


def test_split_sage_preserves_mask_and_grouped_query_heads(monkeypatch):
    calls = []

    def sage(q, k, v, **kw):
        calls.append((q.shape, k.shape))
        return _sage(q, k, v, **kw)

    monkeypatch.setattr(A, "sageattn", sage)
    torch.manual_seed(12)
    q = torch.randn(2, 9, 4, 8)
    k, v = torch.randn(2, 9, 2, 8), torch.randn(2, 9, 2, 8)
    mask = torch.tensor([[True, True, False, False, False], [True, True, True, True, False]])
    params = A.AttentionParams.create_attention_params_from_mask("sageattn", True, 4, mask)
    with torch.no_grad():
        got = A.attention(q, k, v, params)
    for i, n in enumerate((6, 8)):
        want = F.scaled_dot_product_attention(q[i:i+1, :n].transpose(1, 2),
                                              k[i:i+1, :n].transpose(1, 2),
                                              v[i:i+1, :n].transpose(1, 2), enable_gqa=True)
        torch.testing.assert_close(got[i:i+1, :n], want.transpose(1, 2).flatten(2))
        assert torch.count_nonzero(got[i, n:]) == 0
    assert [s[0][2] for s in calls] == [6, 8]
    assert all(qs[1] == 4 and ks[1] == 2 for qs, ks in calls)  # no GQA expansion


def test_unmasked_text_fusion_uses_one_batched_kernel(monkeypatch):
    calls = []

    def sage(q, k, v, **kw):
        calls.append(q.shape)
        return _sage(q, k, v, **kw)

    monkeypatch.setattr(A, "sageattn", sage)
    q = torch.randn(12, 3, 4, 8)
    params = A.AttentionParams.create_attention_params("sageattn", True)
    with torch.no_grad():
        out = A.attention(q, q, q, params)
    assert len(calls) == 1 and calls[0][0] == 12
    torch.testing.assert_close(out, F.scaled_dot_product_attention(
        q.transpose(1, 2), q.transpose(1, 2), q.transpose(1, 2)).transpose(1, 2).flatten(2))


def test_forced_generation_restores_backend_after_failure(monkeypatch, tmp_path):
    path = _prefs(monkeypatch, tmp_path, "1")
    monkeypatch.setattr(A, "sageattn", _sage)
    dit = SimpleNamespace(attn_mode="torch", split_attn=False)

    @G.configured_generation
    def generate(self, model):
        assert (model.attn_mode, model.split_attn) == ("sageattn", True)
        raise RuntimeError("kernel failed")

    with pytest.raises(RuntimeError, match="kernel failed"):
        generate(None, dit)
    assert (dit.attn_mode, dit.split_attn) == ("torch", False)
    monkeypatch.setattr(A, "sageattn", None)
    with pytest.raises(RuntimeError, match="Preferences"):
        generate(None, dit)
    # Changes are read by the next image, including a long-lived training process.
    path.write_text('{"inference_sageattention": "0"}')
    assert G.configured_generation(lambda self, model: model.attn_mode)(None, dit) == "torch"


def test_actual_driver_generation_and_training_keep_separate_backends(monkeypatch, tmp_path):
    from fizgig.families.registry import get
    from fizgig.krea2 import driver as D
    from fizgig.krea2.model import SingleMMDiTConfig, SingleStreamDiT

    path = _prefs(monkeypatch, tmp_path, "0")
    # generate() moves a CPU model's work to the GPU when there is one (block swap streams the weights); this model
    # stays on the CPU, so the check runs CPU-only on a machine with a card too
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(D, "DTYPE", torch.float32)
    monkeypatch.setattr(A, "sageattn", _sage)
    # Never resolve or invoke the CUDA varlen API.
    monkeypatch.setattr(A, "sageattn_varlen", None)
    torch.manual_seed(5)
    model = SingleStreamDiT(SingleMMDiTConfig(features=64, tdim=32, txtdim=24, heads=4, multiplier=2,
                                             layers=1, patch=2, channels=16, txtlayers=3,
                                             txtheads=4, txtkvheads=2)).float().eval()
    driver = get("krea2").load_driver()
    cond = {"hidden_states": torch.randn(8, 3, 24),
            "attention_mask": torch.tensor([True, True, True, False, False, False, True, True])}
    args = dict(steps=2, seed=7, cfg=2., neg_cond=cond)
    baseline = driver.generate(model, cond, 64, 64, **args)
    path.write_text('{"inference_sageattention": "1"}')
    got = driver.generate(model, cond, 64, 64, **args)
    torch.testing.assert_close(got, baseline, atol=2e-5, rtol=2e-5)
    assert (model.attn_mode, model.split_attn) == ("torch", False)
    monkeypatch.setattr(A, "sageattn", lambda *a, **kw: pytest.fail("SageAttention used during training"))
    model.train()
    driver._forward(model, torch.randn(1, 16, 8, 8), torch.tensor([.5]), cond).square().mean().backward()
    assert model.first.weight.grad is not None
