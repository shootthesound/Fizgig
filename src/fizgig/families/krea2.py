"""Krea 2 through the standard layer. It replaced the original Krea 2 trainer (30 Sep 2026) once the two were shown
to train the same: the same model files and Preferences keys, ComfyUI-compatible kohya keys like the original, and
its own cache files (arch_id krea2drv - the cache layout differs from the original's, so its caches are rebuilt).
Facts are the original trainer's (src/fizgig/krea2/trainer.py - removed with the original family on 30 Sep 2026, in git history - utils.py, sampling.py), cited per value.
"""
from fizgig.families.description import (
    GENERAL_NEGATIVE,
    FamilyDescription, LoRAFormat, ModelFile, SamplingSettings, SpeedLoRA,
)

_COMFY = "Comfy-Org/Krea-2"


def _preset(rank, lr=1e-4, adaptive=None, epochs=30, slider=False, mp="0.25"):
    """The original Krea 2 presets' values (lora_trainer_gui.py KREA2_BUILT_IN_PRESETS), in the standard layer's
    keys. adaptive=(min, max) turns Adaptive LR on."""
    lo, hi = adaptive or ("1e-4", "4e-4")
    return {
        "NETWORK_DIM": rank, "NETWORK_ALPHA": rank, "NETWORK_TYPE": "LoRA (standard)", "LEARNING_RATE": lr,
        "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42, "FAMILY_SLIDER": slider,
        "ADAPTIVE_LR": adaptive is not None, "ADAPTIVE_LR_MIN": lo, "ADAPTIVE_LR_MAX": hi,
        "OPTIMIZER_TYPE": "adamw8bit", "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
        "DATASET_MEGAPIXELS": mp, "BLOCKS_SWAP": "Auto (detect from GPU)",
        "FAMILY_PRECISION": "Auto (fits your free VRAM)", "FAMILY_EMA": "0.98 (recommended)",
        "KREA2_LOSS_WATCH": True, "KREA2_PER_IMAGE_LR": True, "KREA2_AUTO_RECAPTION": False,
        "KREA2_WARMUP_LOOK": False,
        # a prompt slider's push strength: 3 on Krea 2 (Peter's slider tests, 1 Oct 2026; Qwen's is 2)
        "FAMILY_SLIDER_GUIDANCE": "3",
    }


KREA2 = FamilyDescription(
    key="krea2",
    arch_id="krea2drv",
    display_name="Krea 2",
    gui_label="Krea 2",
    lora_name_suffix="krea2",
    experimental=True,

    model_files=(
        ModelFile("krea2_raw_dit", "Krea 2 RAW DiT", True, _COMFY, "diffusion_models/krea2_raw_bf16.safetensors",
                  25.9, "The undistilled base the LoRA trains on.", role="dit"),
        ModelFile("krea2_vae", "Qwen-Image VAE", True, _COMFY, "vae/qwen_image_vae.safetensors", 0.25,
                  "16 latent channels, 8x.", role="vae"),
        ModelFile("krea2_text_encoder", "Qwen3-VL-4B text encoder", True, _COMFY,
                  "text_encoders/qwen3vl_4b_fp8_scaled.safetensors", 5.2,
                  "Used for caching only, then unloaded before training steps. fp8_scaled: 3.4 GB less VRAM than "
                  "bf16, virtually identical output.", role="text_encoder"),
        ModelFile("krea2_turbo_lora", "Turbo LoRA (previews)", False, _COMFY,
                  "loras/krea2_turbo_lora_rank_64_bf16.safetensors", 0.47,
                  "8-step previews on the training model.", role="speed_lora", fetch_optional=False),
        ModelFile("krea2_turbo_dit", "Turbo DiT (fp8, workbench previews)", False, _COMFY,
                  "diffusion_models/krea2_turbo_fp8_scaled.safetensors", 13.1,
                  "Repair Studio, LoRA the Explorer and LoRA Royale preview on it by default (8-step, "
                  "CFG-free); without it they use the Turbo LoRA on the RAW model.", role="preview_dit",
                  fetch_optional=False),
    ),
    prefs_note=("💡 Already have these files for ComfyUI? Filling the paths in by hand works perfectly — the download "
                "button is a convenience, not a requirement. The first time you caption or train while online, Fizgig "
                "quietly fetches a few tiny helper files and keeps them, and from then on everything runs fully "
                "offline. Setting up a machine that will never see the internet? Paste a complete HuggingFace model "
                "folder into the text encoder field instead of a single file and nothing needs downloading at all."),
    text_encoder_label="Qwen3-VL-4B",
    vae_label="Qwen-Image VAE",

    latent_channels=16,               # Qwen-Image VAE z_dim 16
    spatial_factor=8,                 # 2 ** len(temperal_downsample)
    bucket_step=16,                   # VAE 8 x patch 2 (the original's RESOLUTION_STEPS)
    image_channels=3,
    native_megapixels=1.0,

    n_blocks=28,                      # SingleStreamDiT: 28 main blocks at width 6144
    block_prefix="blocks",
    block_note="The LoRA also covers the text-fusion stack and the input / timestep / output layers (264 Linears "
               "in all), as the original Krea 2 trainer does.",

    lora=LoRAFormat(
        key_template="lora_unet_blocks_{block}_{module}.{ab}.weight",
        down="lora_down", up="lora_up",
        block_modules=("attn.wq", "attn.wk", "attn.wv", "attn.gate", "attn.wo", "mlp.gate", "mlp.up", "mlp.down"),
        alpha_key="{prefix}.alpha",
        kohya=True,
        file_prefix="lora_unet_",
        note="kohya keys (module path with dots as underscores), as every Fizgig Krea 2 LoRA; ComfyUI reads them.",
        source="src/fizgig/krea2/trainer.py create_network(None, 'lora_unet', ...)",
    ),

    driver="fizgig.krea2.driver:Krea2Driver",
    modelspec_arch="Krea-2",
    implementation="https://github.com/krea-ai/krea-2",
    ema_default="0.98",               # the original's default (Peter's A/B, 9 Sep 2026)
    precisions=("int8", "nf4", "bf16"),
    auto_precisions=("int8", "nf4"),
    compiles=True,
    compile_hint=(
    "Auto (recommended) turns torch.compile on only when this run is long enough to repay it. "
    "It fuses the per-matmul quantise/dequantise work that bounds the INT8 and NF4 paths — "
    "2.0× per step on INT8 (0.59 → 0.29 s/step, matching OneTrainer) and 1.28× on "
    "NF4 (0.71 → 0.56) — but costs a ~90 s compile pause first, so a short run is SLOWER "
    "overall. Break-even is around 600 steps on INT8, 1200 on NF4. NF4 + compile still fits a 16 GB "
    "card (verified under a 13.5 GB cap). INT8 + compile fits from ~22 GB free: at high resolution "
    "the checkpoint automatically moves outside the compiled region, which keeps memory at eager "
    "levels (~18 GB at 1024px, measured ~27% faster than uncompiled). Requires Triton and, on "
    "Windows, a C++ compiler (VS Build Tools) — both located automatically. Never used with "
    "Blocks Swap, since swapping moves weights and compiled graphs assume they stay put."),
    preview_image=True,
    preview_checkpoint_sampling=SamplingSettings(
        "Turbo checkpoint", steps=8, cfg=1.0, sampler="euler", scheduler="simple", options=(("mu", 1.15),),
        note="The distilled Turbo: CFG-free, mu pinned at 1.15, as the original workbench.",
        source="the original Krea 2 workbench engine (removed with the original family)"),
    int8_attention=True,              # workbench renders: comfy-kitchen's INT8 attention
    activation_cache=True,            # Turbo Preview: step-1 replay, identical to a full render
    finetune=True,
    workbench=("repair", "explorer", "profiler", "extract", "royale"),
    # the text-fusion boosts (15 Sep 2026, #137): the four text-fusion blocks at x2 / x3, everything else untouched.
    # Measured across several LoRAs, x3 lifted the detail metric ~72 -> ~77 and likeness 2-7 points on every LoRA but
    # an overtrained one, composition unchanged
    repair_presets=tuple((f"✨Text fusion ×{m} (experimental)",
                          tuple((f"txt_{s}_{i}", float(m)) for s in ("lw", "rf") for i in range(2)))
                         for m in (2, 3)),
    # The original's measured 0.25 MP peaks (utils/capabilities.py: 5090, batch 1, rank 32; 0.42 GB saved per swapped
    # INT8 block) and the driver's measured growth to 1 MP on full-size photos (5090, rank 8, previews off): INT8
    # +2.9 GB (15.3 -> 18.2 GB whole-GPU; the original grows +2.6 on the same data), NF4 13.3 GB at 1 MP under a
    # 16 GB card. The original's +0.25 GB/MP under-plans 1 MP: INT8 + 16 swapped blocks ran out on a 12 GB card.
    # bf16: measured on the driver, rank 8, 0.25 MP, 26.0 GB; its 1 MP point carries INT8's growth.
    train_memory={"int8": (((0.25, 16.2), (1.0, 19.1)), 0.42), "nf4": (((0.25, 11.4), (1.0, 13.4)), 0.0),
                  "bf16": (((0.25, 26.0), (1.0, 28.9)), 0.84)},
    optimizers=("adamw8bit", "adamw"),
    # the original's Automagic setup (Peter, 17 Sep 2026): the 264 Linears converge at very different speeds, and the
    # text-fusion stack (under-trained by the recipe) is the minority one shared rate outvotes; a 16-step sign window
    # because batch one, logit-normal timesteps and bucketed resolutions make consecutive steps different problems,
    # so the default 8 reads that noise as overshoot (measured: the rate crawled 1e-6 -> 4.3e-6 in an epoch)
    optimizer_families=(("txtfusion", ("txtfusion.",)), ("attn", (".attn.",)), ("mlp", (".mlp.",))),
    automagic_sign_window=16,
    network_types=("lora", "lokr"),
    slider_training=True,
    slider_guidance=3.0,              # Peter's slider tests (1 Oct 2026): 3 on Krea 2, where Qwen's 2 pushes too little
    # Ultra mode (Peter, 1 Oct 2026): blocks 0-7 + the four text-fusion blocks. His Repair Studio surgery on slider LoRAs
    # (detail blocks and the I/O layers off) held up at strength 20 without distortion
    slider_ultra_blocks=tuple(f"block_{i}" for i in range(8)) + ("txt_lw_0", "txt_lw_1", "txt_rf_0", "txt_rf_1"),

    sampling=(
        SamplingSettings("RAW", steps=28, cfg=4.5, sampler="euler", scheduler="simple", negative_prompt=True,
                         note="The undistilled model: guided, ~28 steps.", source="src/fizgig/krea2/sampling.py"),
    ),
    speed_loras=(
        SpeedLoRA(
            name="Krea 2 Turbo LoRA (8-step)",
            repo=_COMFY,
            file="loras/krea2_turbo_lora_rank_64_bf16.safetensors",
            pairs_with="Krea 2 RAW",
            strength=1.0,
            settings=SamplingSettings("Turbo 8-step", steps=8, cfg=1.0, sampler="euler", scheduler="simple",
                                      note="mu pinned at 1.15, as the original previews.",
                                      options=(("mu", 1.15),),
                                      source="src/fizgig/krea2/trainer.py _render_prompt_set"),
            load_unmerged=True,
            pref_key="krea2_turbo_lora",
            source=f"https://huggingface.co/{_COMFY}",
        ),
    ),
    preview_speed_lora="Krea 2 Turbo LoRA (8-step)",
    preview_steps=8,
    preview_cfg=1.0,
    preview_negative=GENERAL_NEGATIVE,
    preview_width=1024,
    preview_height=1024,

    presets=(
        ("✨ Krea 2 Ultra Fast (rank 8, adaptive LR)", _preset(8, adaptive=("2e-4", "4e-4"))),
        ("✨ Krea 2 Standard (rank 32, full model)", _preset(32, epochs=64)),
        ("✨ Krea 2 Style (rank 16, gentle LR)", _preset(16, adaptive=("5e-5", "2e-4"), epochs=64)),
        # Slider: hot and short, as Qwen's and the experiment/slider-training branch's Krea 2 Slider (rank 8, 2e-4,
        # 0.5 MP). The loss watch and Adaptive LR are switched off for sliders by the trainer. 20 epochs: a happy/sad
        # prompt slider (16 practice pictures, guidance 2) was right at 20 and had turned +1 into an illustration by 25.
        ("✨ Krea 2 Slider (rank 8, 2e-4)", _preset(8, lr=2e-4, epochs=20, slider=True, mp="0.5")),
    ),
    notes=(
        ("Timesteps: logit-normal with a resolution-dependent shift (mu 0.5 at 256 image tokens to 1.15 at 6400), "
         "unweighted MSE on the flow-matching velocity.", "src/fizgig/krea2/trainer.py sample_krea2_timesteps"),
    ),
)
