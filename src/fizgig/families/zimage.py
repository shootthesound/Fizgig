"""Z-Image Turbo (Tongyi-MAI) through the standard layer: a 6B single-stream DiT (S3-DiT), Qwen3-4B text encoder, the
FLUX.1 VAE. Driver: zimage/driver.py. Experimental.

LoRAs train on Turbo itself with a frozen training adapter (off for previews and never in the saved file): Turbo is
distilled to 8 steps, and a plain LoRA on it undoes the distillation (blurry, washed-out renders at 8 steps).
Sources: the model card, Tongyi's reference code, ComfyUI's implementation, AI-Toolkit's adapter card, Oct 2026.
"""
from fizgig.families.description import FamilyDescription, LoRAFormat, ModelFile, SamplingSettings

_COMFY = "Comfy-Org/z_image_turbo"
_CARD = "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo"
_OSTRIS = "ostris/zimage_turbo_training_adapter"


def _preset(rank, lr=1e-4, adaptive=None, epochs=30, mp="0.5", slider=False):
    # Qwen Image 2.1's preset set (Peter, 7 Oct 2026: "Qwen is a good model for the presets across the various modes"):
    # 0.5 MP, 30 epochs saved every epoch; adaptive=(min, max) turns Adaptive LR on (the run starts at the geometric
    # midpoint and the LR box is ignored), None trains flat at lr. Fused AdamW, as the driver guide asks of LoRA presets.
    lo, hi = adaptive or ("2e-4", "4e-4")
    return {
        "NETWORK_DIM": rank, "NETWORK_ALPHA": rank, "NETWORK_TYPE": "LoRA (standard)", "LEARNING_RATE": lr,
        "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": adaptive is not None, "ADAPTIVE_LR_MIN": lo, "ADAPTIVE_LR_MAX": hi,
        "OPTIMIZER_TYPE": "adamw", "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
        "DATASET_MEGAPIXELS": mp, "BLOCKS_SWAP": "Auto (detect from GPU)",
        "FAMILY_PRECISION": "Auto (fits your free VRAM)", "FAMILY_TRAINING_ADAPTER": True,
        "FAMILY_EMA": "0.98 (recommended)",
        "KREA2_LOSS_WATCH": True, "KREA2_PER_IMAGE_LR": False, "KREA2_AUTO_RECAPTION": False,
        "KREA2_WARMUP_LOOK": False, "FAMILY_FT": False, "FAMILY_SLIDER": slider, "FAMILY_FAST_ID": False,
        "FAMILY_SLIDER_GUIDANCE": "2",
    }


ZIMAGE = FamilyDescription(
    key="zimage",
    arch_id="zimageturbo",
    display_name="Z-Image Turbo",
    gui_label="Z-Image Turbo",
    lora_name_suffix="zit",
    aliases=("z-image", "z-image-turbo", "zimage"),
    experimental=True,

    model_files=(
        ModelFile("zimage_dit", "Z-Image Turbo DiT", True, _COMFY,
                  "split_files/diffusion_models/z_image_turbo_bf16.safetensors", 12.31, role="dit",
                  hint="Z-Image Turbo bf16 (ComfyUI's single file) - trained on and previewed with."),
        ModelFile("zimage_vae", "FLUX.1 VAE", True, _COMFY, "split_files/vae/ae.safetensors", 0.34, role="vae",
                  hint="The FLUX.1 autoencoder (ae.safetensors). Not the FLUX.2 VAE Klein uses."),
        ModelFile("zimage_text_encoder", "Qwen3-4B text encoder", True, _COMFY,
                  "split_files/text_encoders/qwen_3_4b.safetensors", 8.04, role="text_encoder",
                  hint="Used for caching only, then unloaded before training."),
        ModelFile("zimage_training_adapter", "Fizgig training adapter", False,
                  "ShootTheSound/Fizgig-Z-Image-Turbo-Training-Adapter",
                  "fizgig_z_image_turbo_training_adapter.safetensors", 0.07, role="training_adapter",
                  hint="Frozen during training, off in previews and saved LoRAs: keeps Turbo's 8-step look intact "
                       "while the LoRA learns. Against the adapter most people use: better likeness, better image "
                       "quality and better detail."),
    ),
    prefs_title="Model Paths (Z-Image Turbo)",
    text_encoder_label="Qwen3-4B",
    vae_label="FLUX.1 VAE",

    latent_channels=16,               # ae z_channels 16
    spatial_factor=8,
    bucket_step=16,                   # 8x VAE x 2x2 patches (the reference pads tokens to multiples of 32 itself)
    native_megapixels=1.0,            # card: 1024^2 class; ComfyUI template 1024x1024

    n_blocks=30,                      # transformer/config.json n_layers 30 (+ 2 noise and 2 context refiners)
    block_prefix="layers",
    block_note="30 identical single-stream layers (attention + SwiGLU MLP, per-layer AdaLN). The two noise and two "
               "context refiner layers ahead of them are not trained.",
    workbench=("repair", "explorer", "profiler", "extract", "royale"),

    lora=LoRAFormat(
        key_template="diffusion_model.layers.{block}.{module}.{ab}.weight",
        down="lora_A", up="lora_B",
        block_modules=("attention.to_q", "attention.to_k", "attention.to_v", "attention.to_out.0",
                       "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"),
        alpha_key="{prefix}.alpha",
        kohya=False,
        file_prefix="diffusion_model.",
        note="Diffusers module names under diffusion_model., as AI-Toolkit writes Z-Image LoRAs; ComfyUI slices "
             "to_q / to_k / to_v onto its fused attention.qkv and maps to_out.0 to attention.out.",
        source="ComfyUI comfy/lora.py model_lora_keys_unet Lumina2 branch + comfy/utils.py z_image_to_diffusers",
    ),

    driver="fizgig.zimage.driver:ZImageDriver",
    modelspec_arch="Z-Image-Turbo",
    implementation="https://github.com/Tongyi-MAI/Z-Image",
    training_adapter="zimage_training_adapter",
    # Fizgig lab A/B 6 Oct 2026 (170-photo character set, rank 16, 1e-4, 3000 steps, Turbo 8-step renders with the
    # adapter off, ArcFace vs the set): likeness at step 3000 none 54.8, AI-Toolkit v2 72.3, Fizgig 78.1 (ahead from
    # step 1500 on); Fizgig's keeps Turbo's own texture (detail vs the base render 0.97-1.07 on a no-trigger prompt,
    # AI-Toolkit's 1.3-1.45: oversaturated, glossy skin). Recipe as Qwen's adapter: rank 16 on attention + MLP of the
    # 30 layers, 7.6k real photos + 1k of Turbo's own renders (1 step in 3), 2000 steps, EMA.
    training_adapter_note=("Keeps Z-Image Turbo's 8-step look intact: frozen at 1.0 for every training step, off for "
                           "previews and never in your saved file. Without it Turbo LoRAs go soft and painterly."),
    ema_default="0.98",
    precisions=("bf16", "int8", "nf4"),
    # Measured 6 Oct 2026 (RTX PRO 6000, rank 16, fused AdamW, gradient checkpointing, training adapter on, a 1024^2
    # preview each epoch), peak GB including the preview, which sets it - training alone 13.7 / 15.0 (bf16), 8.8 / 10.0
    # (INT8), 7.0 / 8.1 (NF4) at 0.5 / 1 MP; s/step bf16 0.50 / 0.92, INT8 0.53 / 1.04, NF4 0.51 / 0.94. Per swapped
    # layer: its weights (~181M params: 0.36 GB bf16, 0.18 INT8).
    train_memory={"bf16": (((0.5, 17.1), (1.0, 17.2)), 0.36), "int8": (((0.5, 12.3), (1.0, 12.3)), 0.18),
                  "nf4": (((0.5, 10.0), (1.0, 10.1)), 0.0)},
    optimizers=("adamw", "adamw8bit"),
    network_types=("lora", "lokr"),
    slider_training=True,             # photo pairs (training_loss diff weighting) or prompts (noise_latents / predict)
    # full fine-tune (families/ft.py, the driver's ft_spec): attention and MLP of the 30 layers on an NF4 trunk, the
    # training adapter frozen on top as for a LoRA; checkpoints keep ComfyUI's fused-qkv file layout
    finetune=True,
    ft_learning_rate=5e-5,            # the rate ticking Fine-tune sets (see the Fine-tune preset: weights ~3x Qwen's)
    # torch.compile (6 Oct 2026, RTX PRO 6000, 1 MP, rank 16, 20 photos, steady from epoch 3): checkpoint outside the
    # compiled layers - INT8 1.03 -> 0.69 s/step (1.49x, +0.3 GB), bf16 0.92 -> 0.78 (1.18x, +0.1 GB). Inside was a
    # little faster (0.62 / 0.75) but INT8 peaked +10 GB, so outside. Epoch 1 ~1.85 s/step while it compiles, plus a
    # recompile per new bucket shape; payback rounded up from ~50 (INT8) / ~130 (bf16) steps on one bucket.
    compiles=True,
    compile_boundary="outside",
    compile_fullgraph=False,
    compile_payback_steps={"int8": 300, "bf16": 600},
    compile_hint=("Auto (recommended) turns torch.compile on only when this run is long enough to repay it. On Z-Image "
                  "Turbo it is measured 1.49x per step on the INT8 base (1.03 -> 0.69 s/step at 1 MP) and 1.18x on "
                  "bf16 (0.92 -> 0.78), with almost no extra memory: the gradient checkpoint stays outside the compiled "
                  "layers. The first epoch runs slower while the layers compile, so Auto waits for runs longer than "
                  "about 300 steps on INT8 and 600 on bf16; NF4 is not compiled by Auto (On still compiles). Requires "
                  "Triton and, on Windows, a C++ compiler (VS Build Tools) - both located automatically. Never used "
                  "with Blocks Swap, since swapping moves weights and compiled graphs assume they stay put."),
    int8_attention=True,              # workbench renders through attend(); speed-up to be measured
    activation_cache=True,            # Turbo Preview: the 30 layers sit in one ModuleList, on_step every step
    helper_files=(("Tongyi-MAI/Z-Image-Turbo", ("tokenizer/*", "vae/config.json")),),

    sampling=(
        SamplingSettings("Turbo", steps=8, cfg=1.0, sampler="euler", scheduler="simple", options=(("shift", 3.0),),
                         note="8 steps, no CFG, shift 3 (the reference: 9 scheduler steps whose last is skipped).",
                         source=f"{_CARD}; Comfy-Org workflow_templates image_z_image_turbo.json"),
    ),
    preview_steps=8,
    preview_cfg=1.0,
    preview_negative=None,            # CFG-free
    workbench_follows_samples=True,   # the workbench previews as the Samples tab says (as Qwen 2.1)
    samples_cfg_free=True,            # Turbo samples at CFG 1 on a fixed schedule: CFG and the negative grey out
    preview_width=1024,
    preview_height=1024,

    presets=(
        # Qwen Image 2.1's modes, Qwen's values (its notes give the measurements behind each). The adapter A/B trained
        # rank 16 at a flat 1e-4, 1 MP, to likeness 78 by ~3000 steps on 170 photos.
        ("✨ Z-Image Turbo Fast (rank 8, adaptive LR)", _preset(8, adaptive=("2e-4", "4e-4"), mp="0.25")),
        ("✨ Z-Image Turbo Standard (rank 16, adaptive LR)", _preset(16, adaptive=("1e-4", "2e-4"))),
        # Style: flat, because style loss descends steadily and Adaptive LR climbs toward its ceiling there
        ("✨ Z-Image Turbo Style (rank 16, 1.5e-4)", _preset(16, lr=1.5e-4)),
        # Slider: sliders run hot and short (rank 8, 2e-4)
        ("✨ Z-Image Turbo Slider (rank 8, 2e-4)", _preset(8, lr=2e-4, slider=True)),
        # full fine-tune, the training adapter on. Not the shared 1e-5: Z-Image's weights are ~3x larger (median |w| 0.082
        # vs Qwen 2.1 0.024 / Krea 2 0.029) and Adam moves each weight by ~the rate, so 1e-5 is a third of the relative
        # change - measured 7 Oct 2026 (Arcane, 341 frames, 0.5 MP, 10 rotations): 1e-5 left the picture unchanged,
        # 5e-5 learned the style, 1e-4 learned it fully and still clean.
        ("✨ Z-Image Turbo Fine-tune (5e-5)", {**_preset(16, lr=5e-5), "FAMILY_FT": True, "FAMILY_FT_ROTATIONS": "10"}),
        ("✨ Z-Image Turbo Fine-tune Strong (1e-4)", {**_preset(16, lr=1e-4), "FAMILY_FT": True,
                                                       "FAMILY_FT_ROTATIONS": "10"}),
    ),

    notes=(
        ("Train on Turbo with the training adapter active (AI-Toolkit's recipe); a LoRA trained without it breaks "
         "the distillation and renders soft at 8 steps.", f"https://huggingface.co/{_OSTRIS}"),
        ("Text: Qwen3-4B, the chat template with thinking on and the generation prompt, hidden_states[-2], real "
         "tokens only, at most 512.", "Tongyi-MAI/Z-Image src/zimage/pipeline.py"),
    ),
)
