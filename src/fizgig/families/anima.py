"""Anima (CircleStone Labs) through the standard layer: an anime / illustration model, Cosmos-Predict2 2B DiT + LLM
adapter, Qwen3 0.6B text encoder, Qwen-Image VAE. Driver: anima/driver.py.

Facts from the model card, ComfyUI's implementation and the trainers that support it (diffusion-pipe, sd-scripts,
AI-Toolkit, OneTrainer), and community reports (Hugging Face discussions, note.com, the Japanese wikis), Oct 2026.
"""
from fizgig.families.description import FamilyDescription, LoRAFormat, ModelFile, SamplingSettings, SpeedLoRA

_REPO = "circlestone-labs/Anima"
_LORAS = "circlestone-labs/Anima-Official-LoRAs"
_CARD = f"https://huggingface.co/{_REPO}"
_SHIFT = (("shift", 3.0),)


def _preset(rank, lr, epochs=30, mp="1.0", slider=False):
    return {
        "NETWORK_DIM": rank, "NETWORK_ALPHA": rank, "NETWORK_TYPE": "LoRA (standard)", "LEARNING_RATE": lr,
        "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42, "ADAPTIVE_LR": False,
        "OPTIMIZER_TYPE": "adamw", "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
        "DATASET_MEGAPIXELS": mp, "BLOCKS_SWAP": "Auto (detect from GPU)",
        "FAMILY_PRECISION": "Auto (fits your free VRAM)", "FAMILY_EMA": "0.98 (recommended)",
        "KREA2_LOSS_WATCH": True, "KREA2_PER_IMAGE_LR": False, "KREA2_AUTO_RECAPTION": False,
        "KREA2_WARMUP_LOOK": False, "FAMILY_FT": False, "FAMILY_SLIDER": slider,
    }


ANIMA = FamilyDescription(
    key="anima",
    arch_id="anima",
    display_name="Anima",
    gui_label="Anima (experimental)",
    lora_name_suffix="anima",
    aliases=("anima",),
    experimental=True,

    model_files=(
        ModelFile("anima_dit", "Anima Base DiT", True, _REPO, "split_files/diffusion_models/anima-base-v1.0.safetensors",
                  4.18, role="dit",
                  hint="Anima Base v1.0 - the version LoRAs are trained on (the model card's advice). LoRAs trained "
                       "on it also work on Anima Aesthetic and Turbo.",
                  download_note="~4.2 GB - CircleStone Labs (anima-base-v1.0.safetensors), non-commercial licence"),
        ModelFile("anima_vae", "Qwen-Image VAE", True, _REPO, "split_files/vae/qwen_image_vae.safetensors", 0.25,
                  role="vae", hint="The Qwen-Image VAE - the same file Krea 2 uses."),
        ModelFile("anima_text_encoder", "Qwen3 0.6B text encoder", True, _REPO,
                  "split_files/text_encoders/qwen_3_06b_base.safetensors", 1.19, role="text_encoder",
                  hint="Qwen3 0.6B base - used for caching, then unloaded before training."),
        ModelFile("anima_turbo_lora", "Anima Turbo LoRA (previews)", False, _LORAS,
                  "anima-turbo-lora-v0.2.safetensors", 0.15, role="speed_lora",
                  hint="Optional: fast previews (about 10 steps, no CFG)."),
    ),
    prefs_title="Model Paths (Anima)",
    text_encoder_label="Qwen3 0.6B",
    vae_label="Qwen-Image VAE",

    latent_channels=16,               # Qwen-Image VAE z_dim 16
    spatial_factor=8,
    bucket_step=16,                   # 8x VAE x 2x2 patches
    native_megapixels=1.0,            # card: 512^2 to 1536^2; about 1 MP is the sweet spot, ~2 MP breaks

    n_blocks=28,                      # transformer/config.json num_layers 28
    block_prefix="blocks",
    block_note="28 identical DiT blocks (self-attention, cross-attention to the text, MLP). The LLM adapter that "
               "turns the text into the DiT's context is never trained (the model card: it degrades easily).",
    workbench=("repair", "explorer", "profiler", "extract", "royale"),

    lora=LoRAFormat(
        key_template="lora_unet_blocks_{block}_{module}.{ab}.weight",
        down="lora_down", up="lora_up",
        block_modules=("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.output_proj",
                       "cross_attn.q_proj", "cross_attn.k_proj", "cross_attn.v_proj", "cross_attn.output_proj",
                       "mlp.layer1", "mlp.layer2"),
        alpha_key="{prefix}.alpha",
        kohya=True,
        file_prefix="lora_unet_",
        note="kohya keys, as sd-scripts writes Anima LoRAs; ComfyUI maps lora_unet_blocks_N_... to "
             "diffusion_model.blocks.N.... The official LoRAs' PEFT keys (diffusion_model.*.lora_A/B) load too.",
        source="sd-scripts networks/lora_anima.py; ComfyUI comfy/lora.py model_lora_keys_unet",
    ),

    driver="fizgig.anima.driver:AnimaDriver",
    modelspec_arch="anima",
    implementation=_CARD,
    ema_default="0.98",               # as Krea 2, MiniMax H3 and Qwen 2.1 (Peter, 5 Oct 2026)
    precisions=("bf16", "int8", "nf4"),   # 2.1B DiT, 4.2 GB in bf16; the community reports fp8 bases train badly
    # full fine-tune (families/ft.py, the driver's ft_spec): attention and MLP of every block, rotating on an NF4 trunk;
    # the shared default rate 1e-5 (OneTrainer's Anima fine-tune preset uses 1e-6). Not yet measured on Anima.
    finetune=True,
    slider_training=True,             # from photo pairs (training_loss diff weighting) or prompts (noise_latents)
    # Measured 5 Oct 2026 on a 5090 (rank 16, adamw8bit, gradient checkpointing, 1024 previews), peak GB including
    # the preview, which sets it (training alone: bf16 5.1 / 5.7, INT8 3.4 / 4.0, NF4 2.9 / 3.5 at 0.5 / 1 MP); s/step
    # bf16 0.53 / 0.71, INT8 0.74 / 0.81, NF4 0.69 / 0.60. No block swap.
    # Speed (measured 5 Oct 2026, 5090, 1 MP, rank 16, checkpointing on): 8-bit AdamW 741 ms/step -> fused AdamW 619
    # -> fused AdamW + compiled blocks 452 (isolated loop, synced). Through the GUI (real run, 40 photos): 0.71 -> 0.31
    # s/step from epoch 2, peak 5.7 -> 6.1 GB. Compile warm-up ~40 s plus a few s per new shape.
    compiles=True,
    compile_boundary="inside",
    compile_fullgraph=False,
    compile_payback_steps={"bf16": 300},
    compile_hint=("Auto (recommended) compiles Anima's 28 blocks when the run is long enough to repay the warm-up: "
                  "measured about 2.3x faster on a 5090 (0.71 -> 0.31 s/step at 1 MP, with fused AdamW), about 0.4 GB more memory. The first "
                  "steps pause to compile (~40 s, then a few seconds for each new image shape). Auto waits for runs "
                  "over about 300 steps; the INT8 and NF4 bases are not compiled by Auto (On still compiles). "
                  "Requires Triton and, on Windows, a C++ compiler (VS Build Tools) - both located automatically."),
    train_memory={"bf16": (((0.5, 8.9), (1.0, 8.9)), 0.0), "int8": (((0.5, 7.3), (1.0, 7.3)), 0.0),
                  "nf4": (((0.5, 6.6), (1.0, 6.6)), 0.0)},
    optimizers=("adamw", "adamw8bit"),
    network_types=("lora", "lokr"),
    helper_files=(("circlestone-labs/Anima-Base-v1.0-Diffusers",
                   ("tokenizer/*", "t5_tokenizer/*")),),
    workbench_follows_samples=True,   # previews take the Samples tab's steps, CFG and negative
    int8_attention=True,              # workbench renders: comfy-kitchen's INT8 attention (5 Oct 2026, 5090: 1.13x,
    #                                   20 steps 5.04 -> 4.47 s; same composition, small details differ)

    sampling=(
        SamplingSettings("Anima Base", steps=30, cfg=4.5, sampler="euler", scheduler="simple", options=_SHIFT,
                         negative_prompt=True,
                         note="Shift 3 (ComfyUI's default for Anima). Card: 30-50 steps, CFG 4-5, er_sde; the "
                              "community finds 16-20 steps enough and settles on CFG 4.5.",
                         source=f"{_CARD}; ComfyUI supported_models.py Anima (shift 3.0); note.com step sweep"),
    ),
    speed_loras=(
        SpeedLoRA(
            name="Anima Turbo LoRA v0.2",
            repo=_LORAS, file="anima-turbo-lora-v0.2.safetensors",
            pairs_with="Anima Base v1.0",
            strength=1.0,
            settings=SamplingSettings("Turbo", steps=10, cfg=1.0, sampler="euler", scheduler="simple", options=_SHIFT,
                                      note="CFG 1, 8-12 steps (the official card).",
                                      source=f"https://huggingface.co/{_LORAS}"),
            load_unmerged=True,
            pref_key="anima_turbo_lora",
            community_settings=(("strength 0.4-0.6 at 12 steps with CFG 2-3.5 keeps the negative prompt",
                                 "civarchive mirror of the Civitai page"),),
            caveats=("Shortens limbs a little (community).",),
            source=f"https://huggingface.co/{_LORAS}",
        ),
    ),
    preview_speed_lora="Anima Turbo LoRA v0.2",
    # With the Turbo LoRA file set, previews still default to the plain model (strength 0), so its steps are the plain
    # model's 20, not the turbo's own 10
    preview_speed_steps=20,
    preview_speed_strength=0.0,       # previews on the base by default; raise Turbo strength for speed
    preview_steps=20,
    preview_cfg=4.5,
    preview_negative=("worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, "
                      "chromatic aberration"),   # the model card's
    preview_cfg_note="About 4.5 (below 2 washes out, above about 7 oversaturates). Anima reads danbooru tags or "
                     "prose; the card's quality prefix is \"masterpiece, best quality, score_7, safe,\".",
    preview_width=1024,
    preview_height=1024,
    repair_size=1024,                 # a 1 MP model (the card: 512-1536, about 1 MP best)

    presets=(
        # community values (Oct 2026), higher than the card's light-touch 2e-5: characters at rank 8-16 and 1e-4
        # (sd-scripts guides, HF discussions), styles at 5e-5. Not yet measured in Fizgig.
        # 50 epochs: the character guides aim for ~1,000-1,500 steps on 20-30 images; every epoch is saved
        ("✨ Anima Character (rank 16, 1e-4)", _preset(16, 1e-4, epochs=50)),
        ("✨ Anima Style (rank 16, 5e-5)", _preset(16, 5e-5)),
        ("✨ Anima Official (rank 32, 2e-5)", _preset(32, 2e-5)),
        # Slider: Qwen's slider recipe (rank 8, 2e-4, 30 epochs). Not yet measured on Anima
        ("✨ Anima Slider (rank 8, 2e-4)", _preset(8, 2e-4, slider=True)),
        # full fine-tune at the shared fine-tune rate (Peter), and at OneTrainer's
        ("✨ Anima Fine-tune (1e-5)", {**_preset(16, 1e-5), "FAMILY_FT": True, "FAMILY_FT_ROTATIONS": "10"}),
        # OneTrainer's Anima fine-tune rate ("#anima Finetune.json": 1e-6, Adafactor, the
        # transformer blocks, text encoder frozen - the same scope as here)
        ("✨ Anima Fine-tune Official (1e-6)", {**_preset(16, 1e-6), "FAMILY_FT": True,
                                                   "FAMILY_FT_ROTATIONS": "10"}),
    ),

    notes=(
        ("Licence: CircleStone Labs Non-Commercial License (and NVIDIA's Open Model License for the base). LoRAs are "
         "derivatives; generated images may be used commercially.", f"{_CARD}/blob/main/LICENSE.md"),
        ("Do not train the LLM adapter - it degrades easily. Train on Base, not Turbo or Aesthetic.", _CARD),
        ("Text: Qwen3 0.6B last hidden state on the raw caption plus the caption's T5 token ids through the adapter, "
         "padded to 512 tokens. Captions over 512 tokens are cut.", "ComfyUI comfy/text_encoders/anima.py"),
        ("Training: rectified flow, logit-normal timesteps, unweighted MSE (diffusion-pipe, sd-scripts). "
         "noise_offset washes colours out; fp8 bases train badly.", "HF discussions #35, #100"),
    ),
)
