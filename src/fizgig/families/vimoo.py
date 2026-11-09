"""Vimoo 1.0 Lite: Vimoo-AI's ~9.5B text- and image-to-video model (Apache-2.0), one checkpoint for both.

Trains on photos and clips (24 fps, 4n + 1 frames) from the single files Vimoo ships for ComfyUI
(Vimoo-AI/Vimoo-1.0-Lite-ComfyUI): the transformer, Qwen2.5-VL-7B's text tower, the Glyph-ByT5 encoder for quoted
text, SigLIP for image-to-video, Wan 2.2's TI2V VAE and the distilled LoRA that renders in 12 steps. The model code is
in fizgig/vimoo/ (a port of Vimoo's diffusers fork with the ComfyUI file's names, so the LoRA loads in ComfyUI).
"""
from fizgig.training.optimizers import MUON_DEFAULT_ARGS
from fizgig.families.description import (
    ClipSpec,
    FamilyDescription,
    FamilyOption,
    LoRAFormat,
    ModelFile,
    SamplingSettings,
    SpeedLoRA,
    VideoWorkbench,
)

_REPO = "Vimoo-AI/Vimoo-1.0-Lite-ComfyUI"
_MODULES = ("img_attn.qkv", "img_attn.proj", "img_mlp.0", "img_mlp.2",
            "txt_attn.qkv", "txt_attn.proj", "txt_mlp.0", "txt_mlp.2")

# The distilled LoRA's own schedule (its sidecar, vimoo_v1_ti2v_distilled_lora_fp32.safetensors.json): 12 Euler steps on
# these post-shift timesteps, CFG 1
_DISTILLED_SIGMAS = (1.0, 0.9983, 0.9965, 0.9947, 0.9910, 0.9839, 0.9730, 0.9524, 0.9231, 0.8780, 0.8000, 0.6316,
                     0.0)

OPTIONS = (
    # the glyph encoder and SigLIP reach the trainer (previews) and the text cache by path, as H3's audio VAE does
    FamilyOption("VIMOO_GLYPH", "", kind="fixed", tokens="glyph=pref:vimoo_glyph aux:glyph=pref:vimoo_glyph"),
    FamilyOption("VIMOO_SIGLIP", "", kind="fixed", tokens="siglip=pref:vimoo_siglip aux:siglip=pref:vimoo_siglip"),
    FamilyOption("VIMOO_I2V_SHARE", "Image-to-video share",
                 choices=(("Off", ""), ("25% of clip steps", "i2v_share=0.25"), ("50% of clip steps", "i2v_share=0.5")),
                 hint="That share of clip steps trains image-to-video: the clip starts on its own first frame, as "
                      "Vimoo trained text- and image-to-video together. Helps a LoRA meant for image-to-video renders.",
                 show_if_media="clip"),
    FamilyOption("VIMOO_SAMPLE_FRAMES", "Sample length", tab="samples",
                 choices=(("2 seconds (49 frames)", "sample_frames=49"), ("Still (1 frame)", "sample_frames=1"),
                          ("1 second (25 frames)", "sample_frames=25"), ("3 seconds (73 frames)", "sample_frames=73"),
                          ("5 seconds (121 frames)", "sample_frames=121")),
                 hint="Training previews render as clips of this length at 24 fps (a still is quickest)."),
)


def _preset(rank, lr=1e-4, epochs=40, mp="0.25", optimizer="muon", slider=False, ft=False):
    return {
        "FAMILY_SLIDER": slider, "FAMILY_SLIDER_GUIDANCE": "3", "FAMILY_FT": ft,
        "NETWORK_DIM": rank, "NETWORK_ALPHA": rank, "NETWORK_TYPE": "LoRA (standard)",
        "LEARNING_RATE": lr, "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "OPTIMIZER_TYPE": optimizer, "OPTIMIZER_ARGS": MUON_DEFAULT_ARGS if optimizer == "muon" else "",
        "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
        "DATASET_MEGAPIXELS": mp, "FAMILY_EMA": "0.98 (recommended)", "COMPILE_BLOCKS": "Off",
        "BLOCKS_SWAP": "Auto (detect from GPU)", "FAMILY_PRECISION": "Auto (fits your free VRAM)",
    }


VIMOO = FamilyDescription(
    key="vimoo",
    arch_id="vimoo1lite",
    display_name="Vimoo 1.0 Lite",
    gui_label="Vimoo 1.0 Lite",
    lora_name_suffix="vimoo",
    aliases=("vimoo-lite", "vimoo1"),
    experimental=True,
    hidden=False,

    model_files=(
        ModelFile("vimoo_dit", "Vimoo 1.0 Lite", True, _REPO, "diffusion_models/vimoo_v1_lite_bf16.safetensors", 19.3,
                  "Vimoo's ComfyUI single file: one checkpoint for text-to-video and image-to-video.", role="dit",
                  download_note="~19.3 GB - Vimoo-AI/Vimoo-1.0-Lite-ComfyUI -> diffusion_models/"
                                "vimoo_v1_lite_bf16.safetensors"),
        ModelFile("vimoo_vae", "Wan 2.2 TI2V VAE", True, _REPO, "vae/vimoo_v1_wan22_ti2v_vae_fp32.safetensors", 2.82,
                  "Wan 2.2's TI2V VAE (48 channels, 16x spatial, 4x temporal), in Wan's own layout; ComfyUI's "
                  "wan2.2_vae.safetensors is the same file.", role="vae"),
        ModelFile("vimoo_text_encoder", "Qwen2.5-VL-7B (text)", True, _REPO,
                  "text_encoders/vimoo_v1_qwen2_5_vl_7b_bf16.safetensors", 14.1,
                  "Qwen2.5-VL-7B Instruct's text tower (as Qwen Image uses it): loaded for caching and preview "
                  "prompts, then unloaded.", role="text_encoder"),
        ModelFile("vimoo_glyph", "Glyph-ByT5", True, _REPO, "text_encoders/vimoo_v1_glyph_byt5_fp32.safetensors", 0.89,
                  "Glyph-SDXL-v2's ByT5 encoder: reads text in \"double quotes\" in a prompt (signs, titles). Runs on "
                  "the CPU; a prompt without quotes doesn't use it.", role="text_encoder_2", fetch_optional=False),
        ModelFile("vimoo_siglip", "SigLIP so400m", False, _REPO, "clip_vision/vimoo_v1_siglip_so400m_fp16.safetensors",
                  0.86, "SigLIP's vision tower: the reference picture of an image-to-video render.", role="clip_vision",
                  fetch_optional=False),
        ModelFile("vimoo_distilled_lora", "Distilled LoRA (12-step previews)", False, _REPO,
                  "loras/vimoo_v1_ti2v_distilled_lora_fp32.safetensors", 2.27,
                  "Vimoo's distilled adapter for Lite: 12 steps at CFG 1 for previews and the workbench.",
                  role="speed_lora", fetch_optional=False),
    ),
    text_encoder_label="Qwen2.5-VL-7B",
    vae_label="Wan 2.2 TI2V VAE",
    options=OPTIONS,

    media=("photo", "clip"),
    workbench_engine="fizgig.vimoo.workbench:VimooWorkbenchEngine",
    clip_spec=ClipSpec(fps=24, frame_step=4, frame_offset=1, max_frames=241, edge_multiple=32, audio_rate=0,
                       audio_channels=0, note="vimoo/vae.py: frames 4n + 1 (TEMPORAL 4), edges /16 x the 2x2 merge"),
    clip_regimes=(("dial", 12, 1.0), ("confirm", 12, 1.0)),
    video_controls=VideoWorkbench(
        lengths=(("Still (1 frame)", 1), ("17 frames (~0.7s)", 17), ("25 frames (~1s)", 25), ("49 frames (~2s)", 49),
                 ("73 frames (~3s)", 73), ("97 frames (~4s)", 97), ("121 frames (~5s)", 121),
                 ("241 frames (~10s)", 241)),
        default_length="49 frames (~2s)",
        sizes=(480, 512, 576, 640, 704, 720, 768, 832, 960, 1024, 1280), default_size=(832, 480),
        scales=(("Full", 1.0), ("½", 0.5))),
    caption_dropout=True,
    detail_crops=False,               # a crop of a frame isn't a clip
    resolution_stages=False,
    latent_channels=48,
    spatial_factor=16,
    bucket_step=32,                   # 16x VAE x the 2x2 hidden merge
    image_channels=3,
    native_megapixels=0.92,           # 1280 x 720

    n_blocks=40,
    block_prefix="double_blocks",
    block_note="40 dual-stream blocks (double_blocks.N), each with a video and a text stream.",
    lora=LoRAFormat(
        key_template="lora_unet_double_blocks_{block}_{module}.{ab}.weight",
        down="lora_down", up="lora_up",
        block_modules=_MODULES,
        alpha_key="{prefix}.alpha",
        kohya=True,
        file_prefix="lora_unet_",
        note="kohya keys on the ComfyUI file's own module names (double_blocks.N.img_attn.qkv ...), the form "
             "ComfyUI's LoRA loader maps onto the model.",
        source="vimoo_v1_lite_bf16.safetensors' tensor names (1031, matched by vimoo/model.py)"),

    driver="fizgig.vimoo.driver:VimooDriver",
    modelspec_arch="Vimoo-1.0-Lite",
    implementation="https://github.com/Vimoo-AI/Vimoo-1.0",
    ema_default="0.98",
    precisions=("int8", "nf4", "bf16"),
    auto_precisions=("int8", "nf4"),
    # Measured 5 Nov 2026 on an RTX 5090 (rank 8, AdamW, eager, checkpointed, no swap, 2 epochs, the default 2-second
    # 832x480 distilled preview after each). The preview sets every peak: training alone is bf16 18.9 / 19.3, INT8
    # 10.5 / 11.0, NF4 7.1 / 7.5 GB at 0.25 / 1 MP stills, and a 49-frame clip (13 latent frames) INT8 12.2 / 13.4, NF4
    # 8.6 / 9.7 GB at 0.25 / 0.4 MP. Clips sit on the same axis as stills at their token load - latent frames x area
    # (driver.plan_run): 13 x 0.4 MP = 5.2. The per-block saving is a double block's own parameter bytes (227M:
    # 0.42 GiB bf16, 0.21 INT8); NF4 never streams.
    train_memory={"bf16": (((0.25, 28.0), (1.0, 28.0)), 0.42),
                  "int8": (((0.25, 19.5), (1.0, 19.5), (5.2, 19.7)), 0.21),
                  "nf4": (((0.25, 15.9), (1.0, 15.9), (5.2, 16.0)), 0.0)},
    # under 20 GB the trainer parks the model for each preview's decode, so the 5090's decode-beside-the-model peak
    # doesn't apply. Measured 5 Nov 2026 on an RTX 4090 Laptop GPU (16 GB; rank 8, AdamW, no swap, 6 photos + a
    # 33-frame clip at 0.25 MP - the clip, 9 latent frames, the heaviest item at 2.2 MP of tokens - with the default
    # 2-second 832x480 distilled preview): INT8 13.1 GB (3.8 s a step), NF4 9.6 GB (3.0 s). The 5090's clip runs add
    # +1.2 GB from 3.2 to 5.2 MP of tokens, carried onto the 5.2 points. NF4 first there: it leaves a 16 GB card room and steps faster (3.0 s against 3.8), and lower-memory
    # machines are where this model is expected to be used most (Peter, 6 Nov)
    train_memory_small={"int8": (((0.25, 13.1), (2.2, 13.1), (5.2, 14.3)), 0.21),
                        "nf4": (((0.25, 9.6), (2.2, 9.6), (5.2, 10.8)), 0.0)},
    auto_precisions_small=("nf4", "int8"),
    optimizers=("adamw", "adamw8bit", "automagic3"),
    network_types=("lora", "lokr"),
    batch_training=False,
    not_offered={
        "training_maps": "its latents are clips (5-D) even for a photo: the maps draw on a still's latent grid",
        "batch_training": "clips train one a step; stills batching not measured for Vimoo",
        "int8_attention": "measured 6 Nov 2026: mottles plain backgrounds (frame-to-frame flicker +13%, the clip 15% "
                          "away from plain attention) for ~2 s a render",
        "detail_crops": "a crop of a clip's frame isn't a clip",
        "resolution_stages": "clips train at their own size: not measured with stages",
    },
    masked_training=True,
    slider_training=True,             # photo / clip pairs (training_loss diff weighting) or prompts (noise_latents / predict)
    activation_cache=True,            # Reuse unchanged blocks: step-1 replay; generate calls on_step before every step
    preview_park_optimizer=True,      # the optimizer state waits on CPU while previews render (room on 16 GB cards)
    ema_short_run=True,
    finetune=True,                    # the 40 double blocks in six ~3 GB parts (driver ft_spec); clips plan at token load
    ft_learning_rate=1e-5,               # the EMA dropdown's "Short run" (decay sized to the run, as H3's)             # photos only: the dataset gives a mask to a still, never to a clip (as H3)
    loss_watch=True,
    training_maps=False,

    sampling=(
        SamplingSettings("Vimoo Lite", steps=50, cfg=4.0, sampler="euler", scheduler="lognorm", negative_prompt=True,
                         note="Euler on the logit-normal grid (shift 8, std 2.5), CFG 4 with guidance rescale 0.85, "
                              "as Vimoo's inference_defaults.json for Lite text-to-video.",
                         source="Vimoo-AI/Vimoo-1.0-Lite-ComfyUI inference_defaults.json"),
    ),
    speed_loras=(
        SpeedLoRA(name="Distilled LoRA (12-step)", repo=_REPO, file="loras/vimoo_v1_ti2v_distilled_lora_fp32.safetensors",
                  pairs_with="Vimoo 1.0 Lite", strength=1.0,
                  settings=SamplingSettings("Distilled 12-step", steps=12, cfg=1.0, sampler="euler",
                                            scheduler="custom", sigmas=_DISTILLED_SIGMAS,
                                            note="CFG 1, the adapter's own 12 post-shift timesteps.",
                                            source="vimoo_v1_ti2v_distilled_lora_fp32.safetensors.json"),
                  load_unmerged=True, pref_key="vimoo_distilled_lora",
                  source="https://huggingface.co/Vimoo-AI/Vimoo-1.0-Lite-ComfyUI"),
    ),
    preview_speed_lora="Distilled LoRA (12-step)",
    preview_speed_steps=12,
    preview_speed_strength=1.0,
    preview_steps=50,
    preview_cfg=4.0,
    preview_negative=("oversaturated colors, overexposed, static, blurry details, subtitles, style, artwork, painting, "
                      "still image, grayish overall tone, worst quality, low quality, JPEG artifacts, ugly, mutilated, "
                      "extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, "
                      "fused fingers, motionless scene, cluttered background, three legs, many people in the "
                      "background, walking backwards"),
    preview_width=832,
    preview_height=480,
    workbench=("repair", "profiler", "extract", "royale"),
    presets=(("✨ Vimoo 1.0 Lite (rank 16)", _preset(16)),
             ("✨ Vimoo 1.0 Lite Slider (rank 8, 2e-4)", _preset(8, lr=2e-4, epochs=16, slider=True)),
             ("✨ Vimoo 1.0 Lite Fine-tune (1e-5)", _preset(16, lr=1e-5, epochs=10, ft=True))),
    start_lr_bands=(("lora", 3.3e-5, 3e-4), ("finetune", 3.3e-6, 3e-5), ("slider", 6.7e-5, 6e-4)),
)
