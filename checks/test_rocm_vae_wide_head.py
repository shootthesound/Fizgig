"""#383: on AMD ROCm, scaled_dot_product_attention fails or returns wrong values for a head wider than 256 (#179), and
a VAE's mid-block attention is one head of 384-512. Klein, Krea 2, Qwen 2.1 and SDXL went through
modules/wide_attention.py since #179; the VAE shared by MOSS-Image / Flux 3 / Fibo 2, Z-Image's FLUX.1 VAE and
Vimoo's did not, so a MOSS-Image preview on an RX 7800 XT died in the decode (hipErrorProfilerNotInitialized).
Here ROCm is simulated (wide_attention.ROCM on) and each VAE's attention must take the direct path, with the same
result as PyTorch's own call."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fizgig.modules import wide_attention  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src" / "fizgig"


@pytest.fixture
def rocm(monkeypatch):
    """ROCm on, and PyTorch's SDPA refuses a head wider than 256 (as the HIP build fails on it)."""
    monkeypatch.setattr(wide_attention, "ROCM", True)
    real = F.scaled_dot_product_attention

    def sdpa(q, k, v, *a, **kw):
        if q.shape[-1] > wide_attention.FAST_HEAD_DIM:
            raise RuntimeError("CUDA error: profiler is not initialized")
        return real(q, k, v, *a, **kw)

    monkeypatch.setattr(F, "scaled_dot_product_attention", sdpa)
    return real


def _same_as_sdpa(block, x, real, monkeypatch):
    got = block(x)
    monkeypatch.setattr(wide_attention, "ROCM", False)
    monkeypatch.setattr(F, "scaled_dot_product_attention", real)
    want = block(x)
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()


def test_flux3_vae_attention_moss_flux3_fibo2(rocm, monkeypatch):
    from fizgig.flux3.autoencoder import AttnBlock
    torch.manual_seed(0)
    block = AttnBlock(512).eval()
    with torch.no_grad():
        _same_as_sdpa(block, torch.randn(1, 512, 6, 5), rocm, monkeypatch)


def test_vimoo_vae_attention(rocm, monkeypatch):
    from fizgig.vimoo.vae import AttentionBlock
    torch.manual_seed(0)
    block = AttentionBlock(384).eval()
    torch.nn.init.normal_(block.proj.weight, std=0.02)        # zero-initialised: give it something to show
    with torch.no_grad():
        _same_as_sdpa(block, torch.randn(1, 384, 2, 4, 3), rocm, monkeypatch)


def test_zimage_vae_gets_the_rocm_processor(rocm, monkeypatch):
    from diffusers import AutoencoderKL
    from fizgig.zimage import pipeline as P
    import fizgig.utils.hf_cache as hf
    small = dict(block_out_channels=(32, 32), down_block_types=("DownEncoderBlock2D",) * 2,
                 up_block_types=("UpDecoderBlock2D",) * 2, latent_channels=4, norm_num_groups=8,
                 layers_per_block=1)
    monkeypatch.setattr(hf, "cached_snapshot_dir", lambda *a, **k: None)
    monkeypatch.setattr(AutoencoderKL, "from_single_file", classmethod(lambda cls, *a, **k: cls(**small)))
    vae = P.load_vae("vae.safetensors", "cpu")
    procs = vae.attn_processors
    assert procs and all(type(p).__name__ == "_WideHeadVAEProcessor" for p in procs.values())


# VAE attention that may call SDPA directly: narrow heads only
_NARROW = {"minimax/vae.py": "dim_head 64", "minimax/audio_vae.py": "causal, narrow heads"}


def test_no_vae_calls_sdpa_directly():
    """A new VAE file with its own attention goes through wide_head_attention (or is listed above, with why)."""
    bad = []
    for f in SRC.rglob("*.py"):
        rel = f.relative_to(SRC).as_posix()
        if not any(w in f.name for w in ("vae", "autoencoder")) or rel in _NARROW:
            continue
        if "scaled_dot_product_attention(" in f.read_text(encoding="utf-8"):
            bad.append(rel)
    assert not bad, bad
