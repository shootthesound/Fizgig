"""ACE-Step's audio VAE encodes a song in overlapping pieces (#347): the pieces must give the whole-song latents."""
import pytest
import torch

pytest.importorskip("diffusers")
from diffusers import AutoencoderOobleck  # noqa: E402

from fizgig.acestep import vae as V  # noqa: E402


def test_pieces_match_one_pass(monkeypatch):
    torch.manual_seed(0)
    small = dict(V.CONFIG, encoder_hidden_size=8, decoder_channels=8)
    model = AutoencoderOobleck(**small).eval()
    # 3 pieces and a short end, a few hops of overlap - the song's frames must line up exactly
    monkeypatch.setattr(V, "CHUNK", 8 * V.HOP)
    monkeypatch.setattr(V, "OVERLAP", 6 * V.HOP)
    audio = torch.randn(2, 27 * V.HOP + 700) * 0.3
    whole = model.encode(audio[None]).latent_dist
    mean, std = V.posterior(model, audio)
    assert mean.shape == whole.mean.shape and std.shape == whole.std.shape
    # the middle of each piece matches; the receptive field reaches past a few hops, so allow float-level slack only
    torch.testing.assert_close(mean, whole.mean, atol=2e-3, rtol=1e-3)
    short = torch.randn(2, 10 * V.HOP)                                  # one piece: the plain one-pass path
    torch.testing.assert_close(V.posterior(model, short)[0], model.encode(short[None]).latent_dist.mean)
