"""A render LoRA over a layer whose input isn't the LoRA's dtype (H3's AdaLN projections kept fp32 under a bf16
LoRA) adds its update instead of failing with "expected mat1 and mat2 to have the same dtype"."""
import torch

from fizgig.networks.lora import LoRAInfModule, LoHaInfModule


def _fp32_layer():
    torch.manual_seed(0)
    return torch.nn.Linear(16, 8).float()


def test_bf16_lora_on_fp32_layer():
    lin = _fp32_layer()
    x = torch.randn(3, 16)
    base = lin(x)
    mod = LoRAInfModule("lora_unet_x", lin, multiplier=1.0, lora_dim=4, alpha=4)
    torch.nn.init.normal_(mod.lora_up.weight)
    mod.lora_down.to(torch.bfloat16)
    mod.lora_up.to(torch.bfloat16)
    mod.apply_to()
    with torch.no_grad():
        y = lin(x)
        want = base + (mod.lora_up.weight.float() @ mod.lora_down.weight.float() @ x.T).T
    assert y.dtype == torch.float32
    assert torch.allclose(y, want, atol=0.1)


def test_bf16_loha_on_fp32_layer():
    lin = _fp32_layer()
    x = torch.randn(3, 16)
    mod = LoHaInfModule("lora_unet_x", lin, multiplier=1.0, alpha=2, out_dim=8, in_dim=16, r1=2, r2=2)
    for p in mod.parameters():
        torch.nn.init.normal_(p)
    for n in ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"):
        getattr(mod, n).data = getattr(mod, n).data.bfloat16()
    mod.apply_to()
    with torch.no_grad():
        y = lin(x)
    assert y.dtype == torch.float32
