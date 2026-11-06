"""The optional Heal download leaves the released Base and preview slots unchanged."""
from fizgig.families.moss_image import MOSS_IMAGE


def test_heal_is_an_opt_in_base_alternative():
    dit = next(f for f in MOSS_IMAGE.model_files if f.role == "dit")
    assert dit.repo == "Comfy-Org/MOSS-Image"
    assert dit.path == "diffusion_models/moss_image_base_bf16.safetensors"
    assert dit.alt_repo == "KBlueLeaf/MOSS-Image-heal"
    assert dit.alt_path == "comfyui/moss_image_heal_bf16.safetensors"
    assert not dit.fetch_choice  # bulk download keeps the official Base
    assert "moss_image_heal_bf16.safetensors" not in dit.replaces
    preview = next(f for f in MOSS_IMAGE.model_files if f.role == "preview_dit")
    assert preview.path == "diffusion_models/moss_image_sft_int8_convrot.safetensors"
