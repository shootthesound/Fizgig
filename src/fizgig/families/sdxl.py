"""SDXL through the standard layer: any SDXL checkpoint as one .safetensors file (Juggernaut XL by default; Illustrious,
Pony, RealVis and the rest load the same way). The checkpoint carries the UNet, VAE and both CLIP text encoders, so
the VAE and text-encoder rows are optional overrides (ModelFile.inside). Driver: sdxl/driver.py.
"""
from fizgig.families.description import FamilyDescription, FamilyOption, LoRAFormat, ModelFile, SamplingSettings

_JUGG = "RunDiffusion/Juggernaut-XL-v9"
_SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
_IN = ("IN04", "IN05", "IN07", "IN08")
_MID = ("MID",)
_OUT = ("OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05")


def _preset(rank, lr=1e-4, adaptive=None, epochs=20, mp="1.0", alpha=None, slider=False):
    lo, hi = adaptive or ("1e-4", "4e-4")
    return {
        "NETWORK_DIM": rank, "NETWORK_ALPHA": rank if alpha is None else alpha, "NETWORK_TYPE": "LoRA (standard)",
        "LEARNING_RATE": lr, "FAMILY_SLIDER": slider, "FAMILY_FT": False,
        "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": adaptive is not None, "ADAPTIVE_LR_MIN": lo, "ADAPTIVE_LR_MAX": hi,
        "OPTIMIZER_TYPE": "adamw", "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
        "DATASET_MEGAPIXELS": mp, "BLOCKS_SWAP": "Auto (detect from GPU)",
        "FAMILY_PRECISION": "Auto (fits your free VRAM)", "FAMILY_EMA": "0.98 (recommended)",
        # the loss watch with per-image LR and auto-recaption on, except for sliders (Peter, 5 Oct 2026)
        "KREA2_LOSS_WATCH": True, "KREA2_PER_IMAGE_LR": not slider, "KREA2_AUTO_RECAPTION": not slider,
        "KREA2_WARMUP_LOOK": False,
    }


SDXL = FamilyDescription(
    key="sdxl",
    arch_id="sdxl",
    display_name="SDXL",
    gui_label="SDXL (any checkpoint)",
    lora_name_suffix="sdxl",
    aliases=("sdxl", "stable-diffusion-xl"),
    experimental=True,

    model_files=(
        ModelFile("sdxl_checkpoint", "SDXL checkpoint", True, _JUGG, "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
                  7.11, role="dit",
                  hint="Any SDXL checkpoint (.safetensors, the single file ComfyUI loads). Juggernaut XL is a strong "
                       "photographic base; Illustrious, Pony, RealVis and other SDXL fine-tunes work the same way. "
                       "Train on the checkpoint you will use the LoRA with.",
                  download_label="Download Juggernaut XL v9",
                  download_note="~7.1 GB - RunDiffusion (Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors), "
                                "CreativeML OpenRAIL-M",
                  alt_repo="OnomaAIResearch/Illustrious-xl-early-release-v0", alt_path="Illustrious-XL-v0.1.safetensors",
                  alt_label="Download Illustrious XL v0.1", alt_note="~6.9 GB - anime / illustration base"),
        ModelFile("sdxl_vae", "VAE (optional)", False, "stabilityai/sdxl-vae", "sdxl_vae.safetensors", 0.33,
                  role="vae", inside="sdxl_checkpoint", fetch_optional=True,
                  hint="Leave empty to use the checkpoint's own VAE. Set it only to swap in a separate SDXL VAE."),
        ModelFile("sdxl_text_encoder", "Text encoders (optional)", False, role="text_encoder", inside="sdxl_checkpoint",
                  fetch_optional=True,
                  hint="Leave empty: both CLIP text encoders are read from the checkpoint. Set it to another SDXL "
                       "checkpoint to take its text encoders instead."),
    ),
    prefs_title="Model Paths (SDXL)",
    prefs_intro="One SDXL checkpoint is all SDXL needs: its VAE and text encoders come from the same file.",
    text_encoder_label="CLIP-L + CLIP-G",
    vae_label="SDXL VAE",

    latent_channels=4,                # unet/config.json in_channels 4
    spatial_factor=8,                 # vae/config.json: 4 downsampling blocks -> 1/8
    bucket_step=64,                   # SDXL's training buckets are multiples of 64
    native_megapixels=1.0,            # trained at 1024x1024 and its aspect buckets

    n_blocks=11,                      # the 11 attention modules (SDXL block-weight names IN04 ... OUT05)
    block_prefix="unet",              # unused: the driver's block_map names SDXL's own blocks
    block_note="SDXL's attention modules under their block-weight names: input 4, 5, 7, 8, the middle block and "
               "output 0-5 (the resnets and the outer input / output blocks carry no attention and are not trained).",
    extract_presets=(
        ("All Blocks", ()),
        ("Input blocks", tuple((b, 1.0) for b in _IN)),
        ("Middle + output blocks", tuple((b, 1.0) for b in _MID + _OUT)),
    ),
    workbench=("repair", "explorer", "profiler", "extract", "royale"),

    lora=LoRAFormat(
        key_template="lora_unet_{block}_{module}.{ab}.weight",
        down="lora_down", up="lora_up",
        block_modules=("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
                       "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0", "ff.net.0.proj", "ff.net.2"),
        alpha_key="{prefix}.alpha",
        kohya=True,
        file_prefix="lora_unet_",
        lokr_kohya_stems=True,
        note="kohya keys on diffusers module names (lora_unet_down_blocks_1_attentions_0_...), which ComfyUI maps for "
             "SDXL. Community files in the LDM layout (lora_unet_input_blocks_4_1_...) load too (driver alias_flat); "
             "their text-encoder parts (lora_te1_ / lora_te2_) are not used.",
        source="ComfyUI comfy/lora.py model_lora_keys_unet (diffusers keys -> lora_unet_ names)",
    ),

    driver="fizgig.sdxl.driver:SDXLDriver",
    modelspec_arch="stable-diffusion-xl-v1-base/lora",
    implementation="https://github.com/Stability-AI/generative-models",
    ema_default="0.98",               # as Krea 2, MiniMax H3 and Qwen 2.1 (Peter, 5 Oct 2026)
    precisions=("bf16", "int8", "nf4"),   # 2.6B UNet: 5.1 GB in bf16
    # Measured 5 Oct 2026 on a 5090 (Juggernaut v9, rank 16, adamw8bit, gradient checkpointing, 1024 previews), peak
    # GB including the preview, which sets it (training alone: bf16 6.0 / 6.5, INT8 4.0 / 4.4, NF4 3.2 / 3.6 at
    # 0.5 / 1 MP); s/step bf16 0.99 / 1.03, INT8 1.41 / 1.47, NF4 1.32 / 1.17. No block swap.
    # Speed (measured 5 Oct 2026, 5090, 1 MP, rank 16, checkpointing on): the step is launch-bound, not compute-bound.
    # 8-bit AdamW steps each of the LoRA's 1,444 tensors separately (~0.3 s/step): 1493 ms/step -> fused AdamW 1046
    # -> fused AdamW + compiled blocks 629 (6.2 GB either way; isolated loop, synced). Through the GUI (real run, 40
    # photos): 1.03 -> ~0.75 s/step from epoch 2, peak 6.6 GB unchanged; a second 5090 host 0.51-0.57 (the step is
    # CPU-launch-bound, so the host CPU sets it). Compile warm-up ~25 s + a few s per new bucket. Also measured, not
    # adopted: the LoRA in bf16 (~5% once compiled), cuDNN autotuning (no gain), loss watch / Adaptive LR off (~0.05 s).
    compiles=True,
    compile_boundary="outside",
    compile_fullgraph=False,
    compile_payback_steps={"bf16": 200},
    compile_hint=("Auto (recommended) compiles SDXL's 70 transformer blocks when the run is long enough to repay the "
                  "warm-up: measured about 1.4x faster on a 5090 (1.03 -> 0.75 s/step at 1 MP, with fused AdamW) and no extra memory. The "
                  "first steps pause to compile (~25 s, then a few seconds for each new image shape). Auto waits for "
                  "runs over about 200 steps; the INT8 and NF4 bases are not compiled by Auto (On still compiles). "
                  "Requires Triton and, on Windows, a C++ compiler (VS Build Tools) - both located automatically."),
    train_memory={"bf16": (((0.5, 10.3), (1.0, 10.3)), 0.0), "int8": (((0.5, 8.2), (1.0, 8.2)), 0.0),
                  "nf4": (((0.5, 7.4), (1.0, 7.4)), 0.0)},
    optimizers=("adamw", "adamw8bit"),
    network_types=("lora", "lokr"),
    # SDXL's per-epoch loss swings with its uniform timesteps, so Adaptive LR's plateau detector reacts to noise:
    # hidden and never sent (Peter, 5 Oct 2026)
    adaptive_lr=False,
    # Slider LoRAs (prompt pairs, or before/after photo pairs): epsilon prediction works as Concept Sliders' SDXL
    # recipe does - the target is built from the UNet's own predictions (families/train.py)
    slider_training=True,
    # full fine-tune (the driver's ft_spec): the UNet's transformer Linears, text encoders frozen; the shared 1e-5 rate
    finetune=True,
    workbench_follows_samples=True,   # previews take the Samples tab's steps, CFG and negative
    # INT8 attention not used: measured 5 Oct 2026 on a 5090, 2.74 -> 2.72 s per 30-step render (SDXL's renders are
    # launch-bound, not attention-bound), so it would only cost fidelity
    helper_files=((_SDXL, ("model_index.json", "*/config.json", "tokenizer/*", "tokenizer_2/*", "scheduler/*")),),

    sampling=(
        SamplingSettings("Juggernaut (community)", steps=30, cfg=3.0, sampler="dpmpp_2m_sde", scheduler="karras",
                         options=(("sampler", "dpmpp_2m_sde_karras"),), negative_prompt=True,
                         note="DPM++ 2M SDE Karras, 30 steps, CFG 4-5, little or no negative: Fooocus's Juggernaut "
                              "default and the Civitai card. CFG above 6-7 turns skin waxy.",
                         source="lllyasviel/Fooocus presets/default.json; civarchive.com/models/133005; "
                                "rundiffusion.com/juggernaut-xl-rundiffusion-guide"),
        SamplingSettings("Euler (softer)", steps=30, cfg=3.0, sampler="euler", scheduler="normal",
                         options=(("sampler", "euler"),), negative_prompt=True,
                         note="Plain Euler, trailing spacing (starts at full noise): a softer look, less pore detail.",
                         source="rundiffusion.com/prompt-guide-for-juggernaut-xi-and-xii; "
                                "huggingface.co/RunDiffusion/Juggernaut-XL-v9/discussions/4"),
    ),
    preview_steps=30,
    preview_cfg=3.0,                  # Peter, 5 Oct 2026 (the card's range is 3-7, "less is a bit more realistic")
    # Peter's SDXL negative (5 Oct 2026); about 130 tokens, encoded in 77-token chunks as ComfyUI does
    preview_negative=("deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, "
                      "text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, "
                      "morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, "
                      "mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned "
                      "face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, "
                      "extra legs, fused fingers, too many fingers, long neck,mask"),
    preview_cfg_note="SDXL needs CFG: 3 is the default, 3 to 5 the usual range (Juggernaut's skin turns waxy above "
                     "6-7). Above 1 the negative prompt applies.",
    samples_text=(("sampler", "sampler DPM++ 2M SDE, Karras schedule (in ComfyUI: dpmpp_2m_sde / karras)"),),
    preview_width=1024,
    preview_height=1024,
    repair_size=1024,                 # a 1 MP model; a 1024 render at 30 steps took 3.3 s on a 5090

    presets=(
        # The first is a first visit's preset (rank 32 : alpha 16, Peter 5 Oct 2026). Flat LR: SDXL's per-epoch loss swings with its uniform timesteps, so
        # Adaptive LR reacted to noise (Peter, 5 Oct 2026: adaptive preset removed, can come back). The former
        # adaptive "Standard" (rank 16, Adaptive 1e-4..4e-4) had the subject by epoch 2-3 on 115 photos at 1 MP.
        # alpha at half the rank (Peter, 5 Oct 2026: SDXL LoRAs gain from the halved alpha; it also halves the LoRA's
        # output scale, so the same 1e-4 moves it half as far per step)
        # 5e-5 for every SDXL preset (Peter, 5 Oct 2026: clearly better results than 1e-4 in his comparisons)
        ("✨ SDXL Strong (rank 32, alpha 16, 5e-5)", _preset(32, lr=5e-5, alpha=16)),      # the default (Peter)
        ("✨ SDXL Standard (rank 16, alpha 8, 5e-5)", _preset(16, lr=5e-5, alpha=8)),
        # Slider: the default's rank 32 / alpha 16 at 5e-5 (Peter, 5 Oct 2026). Not yet measured on SDXL.
        ("✨ SDXL Slider (rank 32, alpha 16, 5e-5)", _preset(32, lr=5e-5, alpha=16, epochs=30, slider=True)),
        # full fine-tune at the shared fine-tune rate (Peter), and at OneTrainer's
        ("✨ SDXL Fine-tune (1e-5)", {**_preset(32, lr=1e-5, alpha=16), "FAMILY_FT": True, "FAMILY_FT_ROTATIONS": "10",
                                      "KREA2_PER_IMAGE_LR": False, "KREA2_AUTO_RECAPTION": False}),
        # OneTrainer's SDXL fine-tune rate: its "#sdxl 1.0" preset takes the TrainConfig default
        # 3e-6 (modules/util/config/TrainConfig.py), text encoders frozen as here
        ("✨ SDXL Fine-tune Official (3e-6)", {**_preset(32, lr=3e-6, alpha=16), "FAMILY_FT": True,
                                                  "FAMILY_FT_ROTATIONS": "10", "KREA2_PER_IMAGE_LR": False,
                                                  "KREA2_AUTO_RECAPTION": False}),
    ),

    options=(
        # A/B 5 Oct 2026 (Juggernaut v9, 115 photos, 1 MP, rank 16 adaptive, 8 epochs, every checkpoint re-rendered
        # with the same sampler, prompts and seeds): Min-SNR 5 + offset 0.0357 reached the likeness no sooner and
        # wandered at epoch 4; plain MSE locked it from epoch 4. One run each, so off by default, kept as options.
        FamilyOption("SDXL_MIN_SNR", "Min-SNR weighting", choices=(("off", "min_snr=0"), ("γ 5", "min_snr=5")),
                     hint="Evens out how much the noisiest and cleanest steps count. Common in other SDXL trainers; in "
                          "Fizgig's A/B it gave no gain, so it starts off.", section="other"),
        FamilyOption("SDXL_NOISE_OFFSET", "Noise offset", kind="entry", tokens="noise_offset={}", default="0",
                     width=8, hint="0.0357 is the offset SDXL base was trained with; 0 (off) trained as well in "
                                   "Fizgig's A/B.", section="other"),
    ),
    notes=(
        ("Captions and prompts longer than CLIP's 77 tokens are encoded in 77-token chunks side by side, as ComfyUI "
         "does, so nothing is cut off.", "ComfyUI sd1_clip.SDTokenizer chunking"),
        ("Training: DDPM epsilon prediction on SDXL's scaled-linear schedule (betas 0.00085-0.012, 1000 steps), "
         "uniform timesteps, unweighted MSE; size conditioning (time_ids) from the bucket size with no crop.",
         f"{_SDXL} scheduler/scheduler_config.json"),
        ("The VAE runs in fp32: SDXL's VAE overflows in half precision.", "madebyollin/sdxl-vae-fp16-fix card"),
    ),
)
