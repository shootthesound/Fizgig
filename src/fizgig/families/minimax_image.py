"""MiniMax H3 Image (MiniMaxAI/MiniMax-H3-Image, 3 Nov 2026): text-to-image and edits on the H3 DiT without its audio
stream, distilled to 6 CFG-free steps with one output head per step, plus its own image VAE decoder. Model code:
minimax_image/ on top of minimax/ (the DiT, the Qwen3-VL-32B text encoder and the VAE encoder are H3's).

Facts from the model card (https://huggingface.co/MiniMaxAI/MiniMax-H3-Image), its pdd_config.json, the Comfy-Org
repack (https://huggingface.co/Comfy-Org/MiniMax-H3-Image) and ComfyUI's code (comfy/ldm/minimax/, nodes_minimax_h3.py),
cited per value. Measured on the weights (10 Nov 2026, safetensors headers): the DiT's keys and shapes are H3's but the
weights moved a long way from H3 fl2va (|dW| 25-46% of |W| on attention out / fc2), so H3's training adapters are not
used here.
"""
from fizgig.families.description import (FamilyDescription, FamilyOption, LoRAFormat, ModelFile, SamplingSettings,
                                         TextEncoderBuild)

_CARD = "https://huggingface.co/MiniMaxAI/MiniMax-H3-Image"
_COMFY = "Comfy-Org/MiniMax-H3-Image"
_H = "Comfy-Org/MiniMax-H3"
# the checkpoint's 6-step schedule (pdd_config.json sigmas; ComfyUI's simple scheduler on the PDD grid lands on these)
_PDD_SIGMAS = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0)


# Likeness optimization's schedule (Peter, 12 Nov 2026: "this number 6 setup is the exact setup"): won a day of runs on
# the Nicola set (Automagic v4, 0.25 MP) - the Before share starts at the model's own (1 in 6) and climbs to 50 by the
# hinge at epoch 6, alternating with 75; 75 held after. Its face shape came earliest, it scored highest at matched
# framing (AdaFace ViT 0.689, above all but 2 of her 45 photos) and kept the backgrounds clean. The boxes hold these
# numbers behind the row's Advanced toggle; minimax_image/driver.py's defaults are the same.
LIKENESS_DEFAULTS = {"BEFORE": "16.67", "BEFORE_END": "50", "HINGE": "6", "SLIDE": "0", "AFTER": "75",
                     "ALTERNATE": "Before the hinge"}


def _preset(rank, epochs=30, slider=False, mp="0.5", edit=False, likeness=False):
    """Krea 2's presets (families/krea2.py _preset; Peter, 10 Nov 2026: "mirror the Krea 2 presets exactly"), plus the
    two this model needs: Fizgig's training adapter on (without it people turn into caricatures) and an edit flag for
    the Edit preset. Every preset trains on Automagic v3 (Peter, 11 Nov 2026), which starts at 2.8e-4 and finds its own
    rate from there (Peter, 11 Nov 2026: started at 1e-4 it climbed to ~1.8e-4 at once, then trickled down to ~9e-5
    over ~1,200 steps; started at 2.8e-4 it eased down ~20% over 400 steps with no overshoot cut, and the previews
    looked better - Automagic keeps roughly the level it starts from, so the start is the choice), and turns Adaptive LR off, so Krea 2's Ultra Fast preset (an LR choice) isn't here."""
    return {
        "NETWORK_DIM": rank, "NETWORK_ALPHA": rank, "NETWORK_TYPE": "LoRA (standard)", "LEARNING_RATE": 2.8e-4,
        "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42, "FAMILY_SLIDER": slider, "FAMILY_EDIT": edit,
        "ADAPTIVE_LR": False, "ADAPTIVE_LR_MIN": "1e-4", "ADAPTIVE_LR_MAX": "4e-4",
        "OPTIMIZER_TYPE": "automagic3", "OPTIMIZER_ARGS": "", "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
        "DATASET_MEGAPIXELS": mp, "BLOCKS_SWAP": "Auto (detect from GPU)",
        "FAMILY_PRECISION": "Auto (fits your free VRAM)", "FAMILY_TRAINING_ADAPTER": True,
        "FAMILY_EMA": "0.98 (recommended)",
        "KREA2_LOSS_WATCH": True, "KREA2_PER_IMAGE_LR": True, "KREA2_AUTO_RECAPTION": False,
        "KREA2_WARMUP_LOOK": False,
        "FAMILY_SLIDER_GUIDANCE": "3",    # Krea 2's prompt-slider push strength
        # on in Standard only (Peter, 11 Nov 2026), on run six's schedule (12 Nov 2026, LIKENESS_DEFAULTS)
        "MINIMAX_IMAGE_LIKENESS_OPT": "1" if likeness else "",
        **{"MINIMAX_IMAGE_LIKENESS_" + k: v for k, v in LIKENESS_DEFAULTS.items()},
    }


MINIMAX_IMAGE = FamilyDescription(
    key="minimax_image",
    arch_id="minimaxh3image",
    display_name="MiniMax H3 Image",
    gui_label="MiniMax H3 Image",
    lora_name_suffix="mmh3i",
    aliases=("minimax-h3-image", "h3-image"),
    hidden=False,

    # Comfy-Org/MiniMax-H3-Image also ships minimax_h3_image_pruned_w4a8.safetensors (12.6 GB): left out, MiniMax H3's
    # loader takes int8 ConvRot or 16-bit files, not w4a8. The upstream bf16 shards (MiniMaxAI, 66 GB) aren't offered:
    # the pruned bf16 below is the same model in the layout the loader reads.
    model_files=(
        ModelFile("minimax_image_dit", "MiniMax H3 Image DiT", True, _COMFY,
                  "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors", 20.98,
                  "The training base: the pruned int8 file ComfyUI runs, kept int8 on the card (~21 GB), or 4-bit "
                  "on smaller cards.", role="dit",
                  alt_repo=_COMFY, alt_path="diffusion_models/minimax_h3_image_pruned_bf16.safetensors",
                  alt_label="bf16", alt_note="~40 GB pruned bf16 - loads as 4-bit (NF4 / HQQ), no int8"),
        ModelFile("minimax_image_vae", "MiniMax H3 Image VAE", True, _COMFY,
                  "vae/minimax_h3_image_vae_fp16.safetensors", 1.17,
                  "Its own VAE: H3's encoder with a decoder made for pictures. Not the H3 video VAE.", role="vae"),
        ModelFile("minimax_image_text_encoder", "Qwen3-VL-32B text encoder", True, _H,
                  "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", 15.69,
                  "MiniMax H3's text encoder, unchanged (the model card). Used for caching only, then unloaded.",
                  role="text_encoder", same_as="minimax_text_encoder",
                  alt_repo=_H, alt_path="text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors", alt_label="bf16",
                  alt_note="~51.5 GB bf16 - the same conditioning (validated on H3), 3.3x the disk"),
        ModelFile("minimax_image_training_adapter", "Fizgig training adapter", False,
                  "ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter",
                  "fizgig_minimax_h3_image_training_adapter.safetensors", 0.15, role="training_adapter",
                  hint="Frozen during training, off in previews and saved LoRAs: keeps MiniMax H3 Image's faces and "
                       "skin true to the model while the LoRA learns. Without it, people trained on this model turn "
                       "into caricatures."),
    ),
    # measured with scripts/check_te_builds.py against the nvfp4 default (10 Nov 2026, RTX PRO 6000); the int8
    # (convrot) build Comfy-Org also publishes doesn't load in H3's encoder loader (rotated weights)
    text_encoder_builds=(
        TextEncoderBuild("bf16", _H, "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors", 51.51,
                         "600d567f6a9629c8574e8e7041b199bdd9c59a986afa7906910a81919610607d", 0.99997,
                         note="worst token 0.9518"),
    ),
    text_encoder_label="Qwen3-VL-32B",
    licence_link=("MiniMax-H3-Image Community License",
                  "https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE"),
    vae_label="MiniMax H3 Image VAE",

    latent_channels=24,               # vae/config.json latent_channels
    spatial_factor=16,                # 16x spatial downsampling (model card)
    bucket_step=32,                   # 16x VAE x 2x2 patch: each DiT token covers 32x32 pixels (model card)
    native_megapixels=4.0,            # "effective working range of about 4 MP to 20 MP" (model card, Sizes);
    #                                   ComfyUI's conditioning node defaults to 2048x2048

    n_blocks=50,                      # transformer/config.json num_layers
    block_prefix="blocks",
    block_note="Block ids follow MiniMax H3's (h3blk_0-49, h3_rf_0-1): the same DiT layout.",

    lora=LoRAFormat(
        key_template="lora_unet_blocks_{block}_{module}.{ab}.weight",
        down="lora_down", up="lora_up",
        block_modules=("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"),
        alpha_key="{prefix}.alpha",
        kohya=True,
        file_prefix="lora_unet_",
        note="MiniMax H3's kohya keys: ComfyUI's MiniMaxH3Image model has H3's module names.",
        source="families/minimax.py LoRAFormat; ComfyUI comfy/model_base.py MiniMaxH3Image(MiniMaxH3)",
    ),

    driver="fizgig.minimax_image.driver:MiniMaxH3ImageDriver",
    modelspec_arch="MiniMax-H3-Image",
    implementation=_CARD,
    helper_files=(),
    # Fizgig's training adapter (10 Nov 2026), the Qwen / Z-Image recipe on this model: rank 16 on attention + MLP of the
    # 50 blocks, 7.6k real photos + 1k of the model's own 6-step renders (1 step in 3), the 6 distilled noise levels each
    # through its own head, EMA 0.998; cosine 1e-4 to step 500, then flat 1e-4 to step 4000 (the released EMA). Without it
    # a 16-epoch LoRA on a 53-photo set (uniform or 60% clean-end draw) drew caricatures, no face found in later
    # close-ups; with it (Peter's eye, 20 epochs at 0.25 MP, 2e-4 Muon) photographic skin and a clear likeness, and a
    # no-trigger 2048 render kept the base model's quality.
    training_adapter="minimax_image_training_adapter",
    training_adapter_note=("Keeps MiniMax H3 Image true to itself while your LoRA learns: frozen at 1.0 for every "
                           "training step, off for previews and never in your saved file. Without it people trained on "
                           "this model turn into caricatures."),                  # the tokenizer and processor are minimax/'s bundled copies (embedder.py)

    # the checkpoint's own int8 ConvRot codes, or a 4-bit base (NF4, HQQ), as MiniMax H3's loader offers them
    precisions=("int8", "nf4", "hqq"),
    auto_precisions=("int8", "nf4"),
    # Measured 10 Nov 2026 on an RTX 5090 (docs/drivers/STILLS.md section 4): rank 8, Muon, EMA 0.98, the training
    # adapter on, 2 epochs on 4 pictures with a 2048x2048 6-step preview after each. The preview sets the peak at every
    # size: int8 trains at 21.0 GB (0.25 MP) / 21.7 (1 MP) and previews at 22.7; NF4 11.3 / 11.9, preview 12.8.
    # Per swapped block: int8 0.29 GB (1 MP, 10 blocks: 18.7 train / 19.8 preview; the header's 0.387 GB of int8
    # weights less what stays resident); NF4 0.15 GB (1 MP, 10 blocks: 10.3 train / 11.3 preview). HQQ (a manual
    # pick, not in Auto) 1 MP: 17.0 train / 17.3 preview; 0.22 GB per block (10 swapped: 14.8 train / 15.2 preview,
    # RTX PRO 6000, 11 Nov 2026).
    train_memory={"int8": (((0.25, 22.7), (1.0, 22.7)), 0.29), "nf4": (((0.25, 12.8), (1.0, 12.8)), 0.15),
                  "hqq": (((0.25, 17.3), (1.0, 17.3)), 0.22)},
    # GB each extra picture takes: fixed, per MP (int8 training peaks, batch 1 vs 3, previews off, RTX PRO 6000,
    # 10 Nov 2026): 20.7 / 21.2 GB at 0.25 MP, 21.3 / 23.0 at 1 MP. Batching is slower per picture here (1 MP: 17 s
    # an epoch at batch 1, 33 s at batch 3), so Auto never batches (batch_auto stays ())
    batch_gb=(0.05, 0.8),
    precision_labels={"auto": "Auto (fits your free VRAM)", "int8": "int8 · most accurate, needs ~24 GB free",
                      "nf4": "4-bit · fits smaller cards", "hqq": "4-bit HQQ · lower error than 4-bit, slower"},
    optimizers=("automagic3", "adamw", "adamw8bit"),
    trainable_dtype="bf16",           # MiniMax H3's LoRA dtype (its trainer: network.to(bfloat16))
    network_types=("lora", "lokr"),
    ema_default="0.98",
    masked_training=True,
    element_boost=True,               # Peter, 11 Nov 2026: to try; off in every preset (MOSS-Image's Fast / Standard / Turbo have it on)
    slider_guidance=3.0,              # Krea 2's (the presets mirror Krea 2's)
    slider_training=True,             # photo pairs (training_loss diff weighting)
    # prompt sliders off (Peter, 11 Nov 2026): a smiling/frowning prompt slider moved clothes, styling and light, never
    # the expression - with the adapter on or off, on the top 3 or the bottom 2 noise levels. On this model a prompt
    # change reshuffles the whole picture, and the slider learns that. noise_latents / predict stay for the maps
    slider_sources=("pairs",),
    edit_training=True,               # one checkpoint for text-to-image and edits (model card)
    edit_references=16,               # "supports up to 16 reference images" (model card)
    edit_note=("Pairs of an original and its edited version, the same crop and shape - about 40 pairs. Each photo "
               "1 MP or larger; Fizgig resizes them."),
    batch_training=True,              # Batch size (Other Options): pictures packed one after another (forward_packed)
    trigger_attention=True,           # training maps' trigger-word map (driver.trigger_attention_hooks)
    # INT8 attention on: the checkpoint asks for it (every block's comfy_attention.config is comfy_kitchen_int8), so
    # ComfyUI renders with it. Measured 10 Nov 2026 (RTX PRO 6000, 2048 px, 6 steps): Studio vs ComfyUI 27.4 dB with
    # it, 21.8 dB without; no faster (7.32 s vs 7.43 s).
    int8_attention=True,
    # Studio's Auto precision from train_memory (the file is int8 already; the shared rule sizes it as bf16)
    workbench_engine="fizgig.minimax_image.workbench:MiniMaxH3ImageWorkbenchEngine",
    activation_cache=True,            # Reuse unchanged blocks: step-1 replay, identical to a full render
    # Compile Blocks (measured 11 Nov 2026, RTX PRO 6000, 8 epochs, previews off, steady-state s/step off -> on):
    # int8 0.25 MP 0.297 -> 0.286 (1.04x, the warm-up pays back after ~1,500 steps), int8 1 MP 0.614 -> 0.556 (1.10x,
    # ~200 steps), NF4 1 MP 0.593 -> 0.526 (1.13x, ~220 steps). Auto's payback isn't per size, so it takes the presets'
    # 0.25 MP figure; NF4 wasn't timed at 0.25 MP and takes int8's
    compiles=True,
    compile_payback_steps={"int8": 1500, "nf4": 1500},
    # gradient checkpointing off, compiled, no swap, previews on (11 Nov 2026): 23.4 GB at 0.25 MP, 30.6 at 1 MP;
    # 1.5x faster than checkpointed (0.25 MP 0.286 -> 0.19 s/step, 1 MP 0.556 -> 0.38)
    train_memory_nogc={"int8": ((0.25, 23.4), (1.0, 30.6))},
    # the int8 ConvRot Linears stay eager (driver.compile_blocks), so each block's graph breaks around them: no
    # fullgraph, and the checkpoint outside the compiled region (a break can't sit inside a checkpoint in the graph)
    compile_fullgraph=False,
    compile_boundary="outside",
    finetune=False,                   # not built for H3-Image yet (Peter, 11 Nov 2026): no Fine-tune kind until it is
    not_offered={"finetune": "not done for H3-Image yet (Peter, 11 Nov 2026)"},
    ft_learning_rate=3e-5,            # MiniMax H3's tested fine-tune rate (families/minimax.py); to confirm here

    sampling=(
        SamplingSettings("MiniMax-H3-Image (distilled)", steps=6, cfg=1.0, sampler="euler", scheduler="simple",
                         sigmas=_PDD_SIGMAS[:-1],
                         note="6 steps, no CFG, no negative prompt - the API refuses anything else. Best around 16 MP; "
                              "4-20 MP works.",
                         source=f"{_CARD} (Request parameters, Sizes); pdd_config.json"),
    ),
    samples_cfg_free=True,
    samples_text=(
        ("banner", "Preview prompts rendered periodically during training on the MiniMax H3 Image model being "
                   "trained. Samples land in <output_dir>/sample/ and the Gallery button below opens the viewer."),
        ("advanced", "MiniMax H3 Image renders CFG-free in 6 steps, so the knobs it does not use are greyed out."),
        ("neg", "Unused - the model renders without CFG"),
        ("cfg", "The model is distilled to render without CFG"),
        ("steps", "6 = the checkpoint's own schedule, one head per step"),
        ("sampler", "euler, simple")),
    preview_steps=6,
    preview_cfg=1.0,
    preview_negative=None,
    preview_width=2048,               # ComfyUI's MiniMaxH3ImageConditioning default; the low end of 4-20 MP
    preview_height=2048,
    repair_size=1024,
    # Likeness optimization (H3 Image's own, in place of the Clean-end share box, Peter 11 Nov 2026): the share of
    # steps at the noise level below sigma 0.5 (0.232, the last of the 6) follows LIKENESS_DEFAULTS' schedule; the
    # rest spread evenly over the 5 above. Off = the model's own even draw, 1 in 6 (~17%).
    options=(
        FamilyOption("MINIMAX_IMAGE_LIKENESS_OPT", "Likeness optimization", kind="check",
                     tokens="likeness_opt=1", default="1",
                     hint="Weights training toward the last of the model's 6 steps, the nearly finished picture, "
                          "where skin, eyes and fine detail are made, changing over the run so the earlier steps "
                          "(pose, layout, the shape of things) keep up. Untick for H3 Image's own even share.",
                     summary="likeness optimization"),
        # its schedule (Peter, 12 Nov 2026): one share before the hinge, one after, and which side alternates (between
        # the two shares, starting with its own). Inside the tick's row, behind its Advanced toggle (closed by default).
        FamilyOption("MINIMAX_IMAGE_LIKENESS_BEFORE", "Before the hinge", kind="entry", tokens="likeness_before={}",
                     default=LIKENESS_DEFAULTS["BEFORE"], width=6, suffix="% of steps at the clean end, epoch 1 to the hinge",
                     requires="MINIMAX_IMAGE_LIKENESS_OPT", pady=(4, 0), advanced=True),
        # Peter, 12 Nov 2026: the Before share can climb (e.g. the model's own 16.67 up to 50) - the structure-heavy
        # start of a low first share, landing at the hinge like a 50 / 75 alternation
        FamilyOption("MINIMAX_IMAGE_LIKENESS_BEFORE_END", "Rises to", kind="entry", tokens="likeness_before_end={}",
                     default=LIKENESS_DEFAULTS["BEFORE_END"], width=6, suffix="% by the hinge (blank: stays at the Before figure)",
                     requires="MINIMAX_IMAGE_LIKENESS_OPT", pady=(4, 0), advanced=True,
                     hint="The Before share climbs evenly from its figure to this one, reaching it on its last epoch "
                          "before the hinge. While the Before side alternates, only its own epochs climb."),
        FamilyOption("MINIMAX_IMAGE_LIKENESS_HINGE", "Hinge epoch", kind="entry", tokens="likeness_hinge={}",
                     default=LIKENESS_DEFAULTS["HINGE"], width=6, suffix="the last epoch of Before",
                     requires="MINIMAX_IMAGE_LIKENESS_OPT", pady=(4, 0), advanced=True),
        FamilyOption("MINIMAX_IMAGE_LIKENESS_SLIDE", "Slide over", kind="entry", tokens="likeness_slide={}",
                     default=LIKENESS_DEFAULTS["SLIDE"], width=6, suffix="epochs from the Before share to the After share (0: a jump)",
                     requires="MINIMAX_IMAGE_LIKENESS_OPT", pady=(4, 0), advanced=True,
                     hint="Starts after the hinge epoch and reaches the After share on its last epoch; the After "
                          "side's alternating, if picked, starts once the slide ends."),
        FamilyOption("MINIMAX_IMAGE_LIKENESS_AFTER", "After the hinge", kind="entry", tokens="likeness_after={}",
                     default=LIKENESS_DEFAULTS["AFTER"], width=6, suffix="% of steps at the clean end, the hinge to the end",
                     requires="MINIMAX_IMAGE_LIKENESS_OPT", pady=(4, 0), advanced=True),
        FamilyOption("MINIMAX_IMAGE_LIKENESS_ALTERNATE", "Alternate", choices=(       # the first is the default
                         ("Before the hinge", "likeness_alternate=before"),
                         ("After the hinge", "likeness_alternate=after"),
                         ("Off", "likeness_alternate=off")),
                     requires="MINIMAX_IMAGE_LIKENESS_OPT", pady=(4, 0), width=18, advanced=True,
                     hint="The side that alternates switches between the two shares every epoch, starting with its "
                          "own; the other side holds its share. Off: each side holds its own share."),
    ),
    workbench=("repair", "profiler", "extract", "royale"),

    presets=(
        ("✨ MiniMax H3 Image Standard (rank 32, full model)", _preset(32, epochs=30, likeness=True)),
        ("✨ MiniMax H3 Image Style (rank 16)", _preset(16, epochs=64)),
        ("✨ MiniMax H3 Image Edit (rank 8)", _preset(8, epochs=12, edit=True)),
        ("✨ MiniMax H3 Image Slider (rank 8)", _preset(8, epochs=20, slider=True)),
    ),
    start_lr_bands=(("lora", 3e-5, 3e-4), ("slider", 7e-5, 6e-4)),     # Krea 2's, as the presets are its
)
