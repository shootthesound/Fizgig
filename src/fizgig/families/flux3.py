"""Flux 3 Image Klein (Black Forest Labs, released 15 Oct 2026): a 5.6B DiT for text-to-image and editing with up to
10 reference images, Qwen3-VL-4B text encoder, the FLUX.2 autoencoder.

Sources: the model cards (huggingface.co/black-forest-labs/flux-3-image-klein, -base, -comfy) and the reference code
(github.com/black-forest-labs/flux-3-image-klein, Apache-2.0).
"""
from fizgig.families.description import FamilyDescription, LoRAFormat, ModelFile, SamplingSettings, TextEncoderBuild

_COMFY = "black-forest-labs/flux-3-image-klein-comfy"
_CARD = "https://huggingface.co/black-forest-labs/flux-3-image-klein"
_REF = "https://github.com/black-forest-labs/flux-3-image-klein"
_MODULES = ("q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out")   # model.py Block, every Linear


def _preset(rank, lr=1e-4, adaptive=None, epochs=30, mp="0.25", edit=False, slider=False, ft=False):
    # Krea 2's preset values (Peter, 15 Oct 2026: base Flux 3's learning rates and training settings on Krea 2's,
    # not Qwen's): 0.25 MP, adamw8bit, per-image LR on, saved every epoch; adaptive=(min, max) turns Adaptive LR on
    # (the run starts at the geometric midpoint, the LR box is ignored), None trains flat at lr. No training adapter
    # (Flux 3 Base is undistilled). Not yet measured on Flux 3.
    lo, hi = adaptive or ("1e-4", "4e-4")
    return {
        "NETWORK_DIM": rank, "NETWORK_ALPHA": rank, "NETWORK_TYPE": "LoRA (standard)", "LEARNING_RATE": lr,
        "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": adaptive is not None, "ADAPTIVE_LR_MIN": lo, "ADAPTIVE_LR_MAX": hi,
        "OPTIMIZER_TYPE": "adamw8bit", "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
        "DATASET_MEGAPIXELS": mp, "BLOCKS_SWAP": "Auto (detect from GPU)",
        "FAMILY_PRECISION": "Auto (fits your free VRAM)", "FAMILY_EMA": "0.98 (recommended)",
        "KREA2_LOSS_WATCH": True, "KREA2_PER_IMAGE_LR": True, "KREA2_AUTO_RECAPTION": False,
        "KREA2_WARMUP_LOOK": False, "FAMILY_FT": ft, "FAMILY_EDIT": edit, "FAMILY_SLIDER": slider,
        "FAMILY_FAST_ID": False, "FAMILY_SLIDER_GUIDANCE": "3",     # Krea 2's prompt-slider push strength
        **({"FAMILY_FT_ROTATIONS": "10"} if ft else {}),
    }


FLUX3 = FamilyDescription(
    key="flux3",
    start_lr_bands=(("lora", 3e-5, 3e-4), ("finetune", 1.7e-5, 1.5e-4), ("slider", 7e-5, 6e-4)),
    masked_training=True,
    arch_id="flux3klein",
    display_name="Flux 3 Image Klein",
    gui_label="Flux 3 Image Klein",
    lora_name_suffix="f3k",
    aliases=("flux-3-image-klein", "flux3-klein"),

    model_files=(
        ModelFile("flux3_dit", "Flux 3 Image Klein Base DiT", True, _COMFY,
                  "diffusion_models/flux3_image_klein_bf16.safetensors", 11.08,
                  "bf16 Base (undistilled) for training. Files with _sd_ in the name are the distilled model.",
                  role="dit", gated=True),
        ModelFile("flux3_vae", "Flux 3 autoencoder", True, _COMFY, "vae/flux2-vae.safetensors", 0.34,
                  "The FLUX.2 autoencoder, as Klein 9B's.", role="vae", gated=True),
        ModelFile("flux3_text_encoder", "Qwen3-VL-4B text encoder", True, _COMFY,
                  "text_encoders/qwen3vl_4b_bf16.safetensors", 8.88,
                  "Used for caching only, then unloaded.", role="text_encoder", gated=True),
        ModelFile("flux3_distilled_dit", "Flux 3 Image Klein (distilled, previews)", False, _COMFY,
                  "diffusion_models/flux3_image_klein_sd_bf16.safetensors", 11.10,
                  "16 steps, no CFG: fast previews and the workbench.", role="preview_dit", gated=True),
    ),
    # measured 21 Oct 2026 with scripts/check_te_builds.py against qwen3vl_4b_bf16 (the same weights as Krea 2's
    # Qwen3-VL-4B; the fp8 build re-measured 22 Oct: cosine 1.00000, worst token 0.9945); the int8 (convrot) build
    # doesn't load in Flux 3's encoder loader
    text_encoder_builds=(
        TextEncoderBuild("bf16 (Krea 2's copy)", "Comfy-Org/Krea-2", "text_encoders/qwen3vl_4b_bf16.safetensors", 8.88,
                         "36f3ff447ef59201722e8f9ce6020c9819fdcfba6aa2608c4e09b1c0ce114e34", 1.0),
        TextEncoderBuild("fp8 (scaled)", "Comfy-Org/Krea-2", "text_encoders/qwen3vl_4b_fp8_scaled.safetensors", 5.24,
                         "54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094", 1.0),
    ),
    text_encoder_label="Qwen3-VL-4B",
    vae_label="FLUX.2 autoencoder",

    latent_channels=128,              # autoencoder.py LATENT_CHANNELS: 32 channels 2x2-packed
    spatial_factor=16,                # autoencoder.py DOWNSAMPLING_RATIO
    bucket_step=16,                   # pipeline.py latent_shape: sides must be multiples of 16
    native_megapixels=1.0,            # pipeline.py default_size 1024x1024 (references up to 4 MP)

    n_blocks=28,                      # the single blocks; the driver's block_map adds the 3 x 5 per-mode blocks
    block_prefix="single_blocks",
    block_note="Five blocks each for the image, the reference images and the text run before the streams meet, "
               "then 28 shared single blocks. Reference blocks only run on edits.",

    lora=LoRAFormat(
        key_template="lora_unet_single_blocks_{block}_{module}.{ab}.weight",
        down="lora_down", up="lora_up",
        block_modules=_MODULES,
        alpha_key="{prefix}.alpha",
        kohya=True,
        lokr_kohya_stems=True,         # LoKR under the same lora_unet_ names, so ComfyUI's Flux3 q/k/v map applies
        file_prefix="lora_unet_",
        note="kohya keys over the reference module names; ComfyUI maps lora_unet_<path>_q_proj / _k_proj / _v_proj "
             "onto thirds of its fused qkv.",
        source="ComfyUI comfy/lora.py model_lora_keys_unet, Flux3 branch (master, 15 Oct 2026)",
    ),

    driver="fizgig.flux3.driver:Flux3Driver",
    modelspec_arch="Flux.3-image-klein",
    implementation=_REF,
    precisions=("bf16", "int8", "nf4"),
    # training peaks, LoRA rank 16, adamw (15 Oct 2026, RTX 5090). 1.0 MP measured for each precision; 0.25 MP from
    # the measured swap run (9.7 GB = 11.08 weights - 10 x 0.245 swapped + 1.07 overhead) minus the 0.4 GB bf16 grew
    # from 0.25 to 1.0 MP. Per swapped block: a single block's weights (0.245 GB bf16, 0.123 INT8). The preview round
    # sizes itself to the VRAM left (load_preview_checkpoint), so it is not the planning figure.
    train_memory={"bf16": (((0.25, 12.3), (1.0, 12.7)), 0.245), "int8": (((0.25, 8.7), (1.0, 9.1)), 0.12),
                  "nf4": (((0.25, 5.8), (1.0, 6.2)), 0.0)},
    optimizers=("adamw", "adamw8bit"),
    network_types=("lora", "lokr"),
    ema_default="0.98",
    compiles=True,
    compile_boundary="outside",       # as Qwen 2.1's: the checkpoint wraps the compiled block
    # 28 single blocks compiled: 0.31 -> 0.26 s/step (1.19x) at 0.25 MP bf16, ~10 s warm-up per shape (15 Oct 2026)
    compile_payback_steps={"bf16": 600, "int8": 600},
    int8_attention=True,              # workbench renders: 4.55 -> 4.25 s at 1024 (1.07x), 30.4 dB to the bf16 render
    finetune=True,                    # driver.ft_spec: every block's Linears on an NF4 trunk, no streaming
    ft_learning_rate=5e-5,            # the rate ticking Fine-tune sets, as the Fine-tune preset
    slider_training=True,             # photo pairs (training_loss diff weighting) or prompts (noise_latents / predict)
    edit_training=True,               # up to 10 references (pipeline.py MAX_REFERENCES), each its own l index
    edit_references=10,               # the DiT takes them all; the text encoder never sees them
    reference_strength=True,          # the workbench's reference strength + previous-frame latent chain, as Klein's
    edit_note=("Pairs of an original and its edited version, the same crop and shape - about 40 pairs is a good "
               "start (20 at least). Each photo 1 MP or larger; Fizgig resizes them."),
    train_preview_checkpoint=True,    # training previews on the distilled (16 steps, no CFG), as Klein's Distilled
    preview_checkpoint_tick="Use the distilled model for samples (16 steps, no CFG - quick)",
    preview_checkpoint_steps_note="Base samples only - the distilled samples use their own 16 steps",
    workbench_follows_samples=True,   # Base CFG needs the Samples tab's negative (empty = the reference's uncond)

    sampling=(
        SamplingSettings("Reference Base", steps=50, cfg=3.5, sampler="euler", scheduler="simple",
                         options=(("alpha", 6.93),), negative_prompt=True,
                         note="Undistilled: Euler, shifted schedule alpha 6.93, CFG against the empty prompt.",
                         source=f"{_CARD}-base model card; {_REF} src/flux3/pipeline.py MODELS"),
    ),
    preview_checkpoint_sampling=SamplingSettings(
        "Reference distilled", steps=16, cfg=1.0, sampler="euler", scheduler="simple", options=(("alpha", 6.93),),
        note="Flow map: each step jumps to the next time, no CFG.",
        source=f"{_CARD} model card; {_REF} src/flux3/sampling.py denoise_flow_map"),
    preview_steps=50,
    preview_cfg=3.5,
    preview_negative="",              # the reference's unconditional is the empty prompt
    preview_width=1024,
    preview_height=1024,
    presets=(
        # Krea 2's presets at Krea 2's values (families/krea2.py), Ultra Fast and Edit at rank 16, Adaptive LR 1e-4 to 2e-4 (Peter, 15 Oct 2026)
        ("✨ Flux 3 Ultra Fast (rank 16, adaptive LR)", _preset(16, adaptive=("1e-4", "2e-4"), epochs=50)),
        ("✨ Flux 3 Standard (rank 32, full model)", _preset(32, epochs=64)),
        ("✨ Flux 3 Style (rank 16, gentle LR)", _preset(16, adaptive=("5e-5", "2e-4"), epochs=64)),
        # Krea 2 has no edit mode: Ultra Fast's and Standard's settings, at 12 epochs (edit pairs learn fast)
        ("✨ Flux 3 Edit (rank 16, adaptive LR)", _preset(16, adaptive=("1e-4", "2e-4"), epochs=12, edit=True)),
        ("✨ Flux 3 Edit Strong (rank 32) - trickier edits", _preset(32, epochs=12, edit=True)),
        ("✨ Flux 3 Slider (rank 8, 2e-4)", _preset(8, lr=2e-4, epochs=20, slider=True, mp="0.5")),
        # full fine-tune at 5e-5, not Krea 2's 1e-5 (Peter, 15 Oct 2026): Flux 3's block weights are Z-Image's size
        # (median |w| 0.066 vs 0.082), where 1e-5 barely moved them
        ("✨ Flux 3 Fine-tune (5e-5)", _preset(16, lr=5e-5, ft=True)),
    ),
    workbench=("repair", "profiler", "extract", "royale"),
    activation_cache=True,            # Turbo Preview: step-1 replay, identical to a full render

    notes=(
        ("Text conditioning keeps the padding: prompts are padded to a multiple of 80 tokens and the DiT attends "
         "every token, so the cache stores the padded sequence.", f"{_REF} src/flux3/text_encoder.py"),
        ("Training timestep distribution is not published; the driver samples logit-normal shifted by the "
         "sampler's alpha 6.93 until measured otherwise.", "assumption, Flux3Driver README"),
    ),
)
