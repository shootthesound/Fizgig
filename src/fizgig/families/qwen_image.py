"""Qwen Image 2.1: the first family described through FamilyDescription.

Facts from the phase-0 research (26 Sep 2026; full notes in the Desktop fizgig_family_descriptions
RESEARCH_qwen_image_2_1_*.md files). Sources are cited per value. Training entry points stay None until
the family is trained through its driver (qwen_image21/driver.py) by the generic cache + train entry points.
"""
from fizgig.families.description import (
    GENERAL_NEGATIVE, FamilyDescription, LoRAFormat, ModelFile, SamplingSettings, SpeedLoRA,
)

_CARD = "https://huggingface.co/Qwen/Qwen-Image-2.1"
_COMFY = "Comfy-Org/Qwen-Image-2.1"
_VIGGLE = "https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo"
_TEMPLATE = "Comfy-Org workflow_templates templates/image_qwen_image_2_1_t2i.json"
_REDDIT = "r/StableDiffusion 'Qwen Image 2.1 4 Steps Turbo Lora is here by Viggle' (community, Sep 2026)"

def _preset(rank, lr=1e-4, adaptive=None, epochs=30, edit=False, slider=False):
    # 0.5 MP, adamw8bit, 30 epochs saved every epoch. adaptive=(min, max) turns Adaptive LR on (the run starts at
    # the geometric midpoint and the LR box is ignored); None trains flat at lr. Detection runs; per-image LR is
    # off, as in the runs these were measured on.
    lo, hi = adaptive or ("2e-4", "4e-4")
    return {
        "NETWORK_DIM": rank, "NETWORK_ALPHA": rank, "NETWORK_TYPE": "LoRA (standard)", "LEARNING_RATE": lr,
        "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42, "FAMILY_EDIT": edit,
        "FAMILY_SLIDER": slider,
        "ADAPTIVE_LR": adaptive is not None, "ADAPTIVE_LR_MIN": lo, "ADAPTIVE_LR_MAX": hi,
        "OPTIMIZER_TYPE": "adamw8bit", "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
        "DATASET_MEGAPIXELS": "0.5", "BLOCKS_SWAP": "Auto (detect from GPU)",
        "FAMILY_PRECISION": "Auto (fits your free VRAM)", "FAMILY_TRAINING_ADAPTER": True,
        "FAMILY_EMA": "0.98 (recommended)",
        "KREA2_LOSS_WATCH": True, "KREA2_PER_IMAGE_LR": False, "KREA2_AUTO_RECAPTION": False,
        "KREA2_WARMUP_LOOK": False,
        "FAMILY_SLIDER_GUIDANCE": "2",   # a prompt slider's push strength (Krea 2's is 3)
        "FAMILY_FAST_ID": False,
    }


QWEN_IMAGE_21 = FamilyDescription(
    key="qwen_image21",
    arch_id="qwenimage21",
    display_name="Qwen Image 2.1",
    gui_label="Qwen Image 2.1 (experimental)",
    lora_name_suffix="qwen21",
    aliases=("qwen-image-2.1", "qwen21"),
    experimental=True,

    model_files=(
        ModelFile("qwen21_dit", "Qwen Image 2.1 DiT", True, _COMFY,
                  "diffusion_models/qwen_image_2.1_bf16.safetensors", 14.23,
                  "bf16 base for training. ComfyUI's single file ships the MLP pre-fused (gate_up).", role="dit"),
        ModelFile("qwen21_vae", "Qwen Image 2.1 VAE", True, "Qwen/Qwen-Image-2.1",
                  "vae/diffusion_pytorch_model.safetensors", 1.35,
                  "Its own VAE (64 latent channels, 16x, RGBA), the diffusers-format file from the official "
                  "repo. Not the Krea 2 / Qwen-Image VAE.",
                  local_name="qwen_image_2.1_vae_diffusers.safetensors", role="vae"),
        ModelFile("qwen21_text_encoder", "Qwen3-VL-8B text encoder", True, _COMFY,
                  "text_encoders/qwen3vl_8b_bf16.safetensors", 17.53,
                  "Used for caching only, then unloaded before training steps.", role="text_encoder"),
        ModelFile("qwen21_training_adapter", "Fizgig training adapter", False,
              "ShootTheSound/Fizgig-Qwen-Image-2.1-Training-Adapter",
              "fizgig_qwen_image_2.1_training_adapter.safetensors", 0.08,
              "Frozen during training, off in previews and saved LoRAs. Without it Qwen 2.1 LoRAs collapse or "
              "wobble; with it likeness was 77 vs 56 in Fizgig's A/B.", role="training_adapter"),
    ModelFile("qwen21_turbo_lora", "Viggle turbo LoRA (previews)", False,
                  "Viggle/Qwen-Image-2.1-viggle-turbo",
                  "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors", 0.68,
                  "Optional: fast in-training previews. Applied unmerged.", role="speed_lora"),
    ),
    text_encoder_label="Qwen3-VL-8B",
    vae_label="Qwen Image 2.1 VAE",

    latent_channels=64,               # vae/config.json z_dim 64
    spatial_factor=16,                # vae/config.json scale_factor_spatial 16
    bucket_step=32,                   # VAE 16 x 2x2 image-pad slot grouping (ai-toolkit bucket divisibility)
    image_channels=4,                 # RGBA in/out; RGB training images get an opaque alpha channel
    native_megapixels=4.0,            # "natively supports 2K" (GitHub README aspect table)

    n_blocks=32,                      # transformer/config.json num_layers 32, identical single-stream blocks
    block_prefix="transformer_blocks",
    block_note="Modulation is shared across all blocks (one global Linear), so per-block sliders act "
               "on attention and MLP only. Identity sits in blocks 10-14 (measured 2 Oct 2026 with the "
               "Profiler on two character LoRAs: those five alone give 67-89% of the likeness, leaving them "
               "out removes 82-84%, block 12 the strongest); the rest shape the picture.",

    # Measured, not guessed (2 Oct 2026, Profiler group + single-block switch-offs, jadelyn / lara character LoRAs,
    # ArcFace vs the subjects' photos): blocks 10-14 carry the identity (alone 67% / 89% of the likeness, left out
    # -84% / -82%), 5-9 a little on one LoRA (11-15%), everything else shapes the picture. Bleed sits in 10-14 too.
    # 15-19 alone gave lara 18% (left out: nothing) - the next place to look if 10-14 ever can't carry a face alone.
    block_categories=tuple((f"block_{i}", "identity" if 10 <= i <= 14 else "look") for i in range(32)),
    # Fast Identity Mode (2 Oct 2026, Sydney, 119 photos, 0.25 MP, 30 epochs, ArcFace vs her photos): blocks 10-14
    # alone 2.90 it/s vs 1.84 for every block (+58%), likeness .507/.587/.620 at epochs 10/20/30 vs .490/.596/.556.
    # One seed per run; Lara (0.5 MP, 15 epochs) matched at epoch 10 and trailed at 15.
    # torch.compile (2 Oct 2026, 5090, 40 photos, 0.25 MP, rank 8, checkpoint outside the compiled blocks - no extra
    # memory): INT8 2.00 -> 2.86 it/s (+43%, settled by epoch 2), bf16 2.04 -> 2.33 (+14%, still rising at epoch 3);
    # epoch 1 ~0.7 it/s while the blocks compile. Payback ~200 / ~400 steps measured, rounded up.
    compiles=True,
    compile_boundary="outside",
    compile_fullgraph=False,
    compile_payback_steps={"int8": 300, "bf16": 800},
    compile_hint=("Auto (recommended) turns torch.compile on only when this run is long enough to repay it. On Qwen "
                  "Image 2.1 it is measured 1.43x per step on the INT8 base (2.00 -> 2.86 it/s) and about 1.15x on "
                  "bf16 (2.04 -> 2.33), with no extra memory: the gradient checkpoint stays outside the compiled "
                  "blocks. The first epoch runs slower while the blocks compile, so Auto waits for runs longer than "
                  "about 300 steps on INT8 and 800 on bf16; NF4 is not compiled by Auto (On still compiles). Requires "
                  "Triton and, on Windows, a C++ compiler (VS Build Tools) - both located automatically. Never used "
                  "with Blocks Swap, since swapping moves weights and compiled graphs assume they stay put."),
    identity_blocks=tuple(f"block_{i}" for i in range(10, 15)),
    repair_presets=(
        ("✨Identity only", tuple((f"block_{i}", 1.0 if 10 <= i <= 14 else 0.0) for i in range(32))),
        ("✨Look only (no identity)", tuple((f"block_{i}", 0.0) for i in range(10, 15))),
        ("✨Identity ×0.5", tuple((f"block_{i}", 0.5) for i in range(10, 15))),
    ),

    lora=LoRAFormat(
        key_template="transformer.transformer_blocks.{block}.{module}.{ab}.weight",
        down="lora_A", up="lora_B",
        block_modules=("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                       "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out"),
        alpha_key="{prefix}.alpha",
        kohya=False,
        file_prefix="transformer.",
        note="Approved exception to the kohya-key rule (Peter, 26 Sep 2026): ComfyUI maps "
             "gate_layer/proj onto the halves of its fused gate_up only for bare or transformer.-prefixed "
             "keys; lora_unet_ / diffusion_model. keys silently drop the MLP input on the fused checkpoint.",
        source="ComfyUI comfy/lora.py model_lora_keys_unet QwenImage branch (commit 6bfaacc67c); "
               "Viggle r128 header",
    ),

    driver="fizgig.qwen_image21.driver:QwenImage21Driver",
    modelspec_arch="Qwen-Image-2.1",
    training_adapter="qwen21_training_adapter",
    ema_default="0.98",               # same default as Krea 2 and MiniMax H3 (measured there, 9 Sep 2026)
    training_adapter_note=("Keeps Qwen 2.1 LoRA training stable: frozen at 1.0 for every training step, off for "
                           "previews and never in your saved file. Without it Qwen 2.1 LoRAs collapse or wobble "
                           "(likeness 77 with it vs 56 without in Fizgig's A/B)."),
    implementation="https://github.com/QwenLM/Qwen-Image",
    precisions=("bf16", "int8", "nf4"),
    # Measured 27 Sep 2026 on a 5090, gradient checkpointing, VAE resident for previews, 1 epoch of 52 steps:
    #   ~1 MP buckets, rank 32: bf16 19.0 GB 2.04 s/step; INT8 11.9 GB 1.94; NF4 9.5 GB 2.02;
    #     bf16 + 16 swapped 11.9 GB 3.38 s/step; INT8 + 16 swapped 8.8 GB 2.50.
    #   0.25 MP (the Fast presets), rank 8 / 16: bf16 14.6 / 14.9 GB; INT8 8.3 / 8.6; NF4 5.8 / 6.0; all ~0.5 s/step.
    # Quantising costs no speed, swapping costs 25-65%, so Auto quantises before it swaps. Figures: (megapixels,
    # peak GB) points - the rank-16 peak at 0.25 MP - and GB saved per swapped block (weights, resolution-free).
    train_memory={"bf16": (((0.25, 14.9), (1.0, 19.0)), 0.44), "int8": (((0.25, 8.6), (1.0, 11.9)), 0.19),
                  "nf4": (((0.25, 6.0), (1.0, 9.5)), 0.0)},
    optimizers=("adamw", "adamw8bit"),
    network_types=("lora", "lokr"),
    edit_training=True,             # one checkpoint for text-to-image and edits (up to 10 references)
    slider_training=True,
    workbench_follows_samples=True,   # Peter, 1 Oct 2026: the workbench previews as the Samples tab says
    # the reference default is CFG 1 (none); community reports find a little CFG helps 2.1 (1.5-3), higher
    # oversaturates (Comfy-Org/Qwen-Image-2.1 discussions, comfyui-wiki, Oct 2026)
    preview_cfg_note="1 = no CFG (the reference default). A little CFG, about 1.5 to 3, gives better previews for "
                     "many people; higher tends to oversaturate. Above 1 the negative prompt applies.",
    int8_attention=True,              # workbench renders: comfy-kitchen's INT8 attention
    activation_cache=True,            # Turbo Preview: step-1 replay, identical to a full render
    finetune=True,                  # the driver's ft_spec (families/ft.py)
    # measured 28 Sep 2026: 40-48 pairs learned a grade on held-out photos in 6-8 epochs at 0.5 MP
    edit_note=("About 40 pairs (20 at least; more if your photos vary a lot). Each photo 1 MP or larger, e.g. "
               "1200x800; bigger is fine, Fizgig resizes them. An original and its edited version must have the "
               "same crop and shape."),
    helper_files=(("Qwen/Qwen-Image-2.1", ("processor/*",)),),   # tokenizer + image processor + chat template

    sampling=(
        SamplingSettings("ComfyUI template", steps=25, cfg=1.0, sampler="euler", scheduler="simple",
                         note="1 MP default; model shift 0.69 (mu) built in, no ModelSampling node.",
                         source=_TEMPLATE),
        SamplingSettings("Official (diffusers)", steps=40, cfg=1.0, sampler="euler", scheduler="simple",
                         note="FlowMatch Euler, dynamic exponential shift (mu 0.5 at 256 tokens to 0.9 at 8192), "
                              "shift_terminal 0.02. 'Meant to be sampled without guidance.'",
                         source=f"{_CARD} + diffusers pipeline_qwenimage21.py"),
        SamplingSettings("Community quality", steps=40, cfg=3.0, sampler="euler", scheduler="simple",
                         negative_prompt=True,
                         note="Community claim: CFG ~3-3.5 with 30-50 steps is more coherent than CFG 1; "
                              "25 steps shows banding that is gone at 35-40.",
                         source=f"{_REDDIT}; HF discussion threads"),
    ),
    speed_loras=(
        SpeedLoRA(
            name="Viggle turbo v0.2.1 (6-step)",
            repo="Viggle/Qwen-Image-2.1-viggle-turbo",
            file="Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors",
            pairs_with="Qwen Image 2.1 base transformer",
            strength=1.0,
            settings=SamplingSettings("Viggle 6-step", steps=6, cfg=1.0, sampler="euler", scheduler="simple",
                                      sigmas=(1.0, 0.9375, 0.875, 0.75, 0.5, 0.25),
                                      note="shift_terminal must be null; 8 steps for small text. "
                                           "Add or remove steps only at the high-noise end.",
                                      options=(("shift_terminal", None),),
                                      source=_VIGGLE),
            load_unmerged=True,
            pref_key="qwen21_turbo_lora",
            community_settings=(
                ("8-20 steps at strength 0.3-0.8 as a clean-up rather than a speed-up; CFG 1-2 works "
                 "when the strength is lowered", _REDDIT),
                ("strength ~0.3 removes the grid pattern almost completely", _REDDIT),
            ),
            caveats=("Grid pattern at 4 steps and full strength (community).",
                     "Weaker for editing than for text-to-image (community).",
                     "Merged loading keeps only part of it (LPIPS 0.093 merged vs 0.052 unmerged)."),
            source=_VIGGLE,
        ),
    ),
    # Previews on the live training DiT: with the Viggle turbo LoRA when its file is set, else the template's 25
    # steps (Krea 2 pattern: live training model, family turbo LoRA, no model swap). The turbo runs at its own
    # 1.0 / 6 steps (Peter, 28 Sep).
    preview_speed_lora="Viggle turbo v0.2.1 (6-step)",
    # Previews default to the plain model (25 steps, turbo strength 0 = not loaded); strength 1 at 6 steps is the
    # fast option.
    preview_speed_steps=25,
    preview_speed_strength=0.0,
    preview_steps=25,
    preview_cfg=1.0,
    preview_negative=GENERAL_NEGATIVE,
    preview_width=1024,
    preview_height=1024,

    presets=(
        # Peter, 27 Sep 2026, from a same-dataset comparison re-rendered identically (Desktop qwen_optimizer_ab/
        # rerender). All 0.5 MP: quicker than 1 MP and keeps more of Qwen's sharpness than 0.25 MP. The first entry
        # is what a first visit to the family applies.
        # Fast: rank 8, Adaptive 2e-4..4e-4 - fastest to likeness (44 at epoch 3 vs 23-31 for every other run) and
        # the best-held skin detail at 0.5 MP. Automagic (lower likeness, softer late) and flat 1e-4 (too slow)
        # both lost to it.
        ("✨ Qwen 2.1 Fast (rank 8, adaptive LR)", _preset(8, adaptive=("2e-4", "4e-4"))),
        # Fast Identity Mode: Fast's recipe on the identity blocks only, at the 0.25 MP it was measured at (Sydney,
        # see identity_blocks above): about 1.5x faster, very close to full-model likeness.
        ("✨ Qwen 2.1 Fast Identity Mode (rank 8) - very close to full-model likeness, ~1.5x faster",
         {**_preset(8, adaptive=("2e-4", "4e-4")), "DATASET_MEGAPIXELS": "0.25", "FAMILY_FAST_ID": True}),
        # Standard: rank 16 for bigger or mixed datasets. Fast's range at rank 16 overcooked from ~epoch 15 (skin
        # detail 6.4 -> 4.7 by epoch 30), so the range is halved; Peter has run this at rank 16.
        ("✨ Qwen 2.1 Standard (rank 16, adaptive LR)", _preset(16, adaptive=("1e-4", "2e-4"))),
        # Style: flat, because style loss descends steadily and Adaptive LR climbs toward its ceiling on steady
        # descent, where style overbakes. 1.5e-4 sits between flat 1e-4 (slow at rank 16) and the range that
        # overcooked rank 16. 30 epochs, every one saved: stop early or pick an epoch in LoRA Royale.
        ("✨ Qwen 2.1 Style (rank 16, 1.5e-4)", _preset(16, lr=1.5e-4)),
        # Edit: Peter's proven recipe (29 Sep, 40-pair Sedona film grade, v6.6.0 demo) - Fast's rank 8 + Adaptive
        # 2e-4..4e-4, best at epoch 9. 12 epochs covers that with headroom; pick the epoch in LoRA Royale.
        ("✨ Qwen 2.1 Edit (rank 8, adaptive LR)", _preset(8, adaptive=("2e-4", "4e-4"), epochs=12, edit=True)),
        # Edit Strong: rank 16 for trickier edits, at Standard's halved range (rank 16 at Fast's overcooked).
        ("✨ Qwen 2.1 Edit Strong (rank 16, adaptive LR) - trickier edits",
         _preset(16, adaptive=("1e-4", "2e-4"), epochs=12, edit=True)),
        # Slider (29 Sep): sliders run hot and short (rank 8 - Peter, 7 Oct - at 2e-4, a few hundred
        # steps). Measured on Qwen: a prompt smile slider was clear by 160 steps, a 40-pair warmth slider by 160.
        ("✨ Qwen 2.1 Slider (rank 8, 2e-4)", _preset(8, lr=2e-4, epochs=30, slider=True)),
    ),
    workbench=("repair", "explorer", "profiler", "extract", "royale"),

    notes=(
        ("A plain LoRA is unstable: collapse to texture at lr 5e-4 (~step 300), wobble at 1e-4, a no-adapter "
         "Adaptive-LR run fell to 37 likeness at step 2000. Loss does not show it. Train with the frozen "
         "training adapter: 77.4 likeness (Fizgig adapter) vs 76.4 (SimpleTuner v2) vs 55.8 (none), and far "
         "better detail with Fizgig's (Peter's eye). Adaptive LR 1e-4..2e-4; ~2750 steps on a 170-image set.",
         "Fizgig lab A/B 26 Sep 2026 (Desktop qwen_ab_results); SimpleTuner assistant-v2 card"),
        ("Text encoder conditioning is the LAST layer BEFORE the final RMSNorm, with a fixed system-prompt "
         "template whose tokens are dropped. transformers >= 5 returns the normed state unless hooked.",
         "diffusers pipeline_qwenimage21.py L206-310"),
        ("Block-causal attention (text causal, each image bidirectional) and t=0 modulation for the text "
         "prefix (causal_condition) are required in every training forward.",
         "diffusers transformer_qwenimage21.py L238-306"),
        ("Gradient checkpointing with only block Linears LoRA-wrapped records no graph unless the joint "
         "hidden states require grad; layer offload produced NaN grads in ai-toolkit.",
         "ostris/ai-toolkit#1054"),
    ),
)
