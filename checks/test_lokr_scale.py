"""#304: a full-matrix LoKR (lokr_w1 / lokr_w2, no _a/_b factors) plays at scale 1 whatever alpha the file stores, as
LyCORIS and ComfyUI do (LyCORIS writes the network dim there); a decomposed factor keeps alpha / its rank."""
import torch

from fizgig.networks.lora import lycoris_scale_from_keys


def test_full_matrix_lokr_scale_is_one():
    full = {"lokr_w1": torch.zeros(4, 4), "lokr_w2": torch.zeros(8, 8), "alpha": torch.tensor(16.0)}
    assert lycoris_scale_from_keys(full) == 1.0
    assert lycoris_scale_from_keys({**full, "alpha": torch.tensor(1.0)}) == 1.0          # Fizgig's own files


def test_decomposed_lokr_keeps_alpha_over_rank():
    dec = {"lokr_w1": torch.zeros(4, 4), "lokr_w2_a": torch.zeros(8, 8), "lokr_w2_b": torch.zeros(8, 8),
           "alpha": torch.tensor(16.0)}
    assert lycoris_scale_from_keys(dec) == 2.0
