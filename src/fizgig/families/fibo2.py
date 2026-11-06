"""Fibo 2.0: a Bria-style diffusion transformer trained as a LoRA over the upstream diffusers repo.

The training base (models--briaai--Fibo-2.0-dev/transformer/) is the diffusers BriaFibo2Transformer2DModel in
its own key layout - the driver (fibo2/driver.py) takes the sharded bf16 checkpoint as it is, and the LoRA is
trained in the split diffusers layout. export_lora_state_dict re-fuses the trained LoRA to ComfyUI's layout
so it loads in ComfyUI over the checkpoint.
"""
# the canonical model repo the family trains from: briaai/Fibo-2.0-dev, the upstream diffusers layout - the
# sharded bf16 DiT (transformer/), the VAE and the text encoder all carry diffusers' own keys, so the family
# needs no remap on load. The paths below are inside that repo; a computer's own copy lives wherever its
# Preferences point (the Hub cache under models--briaai--Fibo-2.0-dev/snapshots/...).
_CANON = "briaai/Fibo-2.0-dev"

from fizgig.training.optimizers import MUON_DEFAULT_ARGS
from fizgig.families.description import (
    FamilyDescription,
    LoRAFormat,
    ModelFile,
    SamplingSettings,
    SpeedLoRA,
    TextEncoderBuild,
)

_MODULES = ("to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3")   # every Linear the LoRA targets

def _preset(rank, lr=1e-6, adaptive=None, epochs=40, mp="0.25", slider=False, edit=False, ft=False, optimizer="muon"):
    """The built-in presets' values, in the standard layer's keys (the shape krea2.py and moss_image.py use). Automagic
    starts at 1e-6 and sets its own rate (measured 4 Nov 2026, rank 32 Full Model, 16 photos at 0.25 MP: up to 1.1e-4 by
    epoch 10, easing to 2.5e-5 by 60; likeness earlier than AdamW at a flat 1e-4, best by eye at epoch 40). A 1e-4 start overcooked the LoRA (PR #250).
    adaptive=(min, max) turns Adaptive LR on."""
    lo, hi = adaptive or ("1e-5", "4e-4")
    return {
        "NETWORK_DIM": rank, "NETWORK_ALPHA": rank, "NETWORK_TYPE": "LoRA (standard)",
        "FAMILY_EDIT": edit, "FAMILY_SLIDER": slider, "FAMILY_FT": ft,
        "FAMILY_SLIDER_GUIDANCE": "3",                               # Krea 2 and Flux 3's prompt-slider push
        "LEARNING_RATE": lr, "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": adaptive is not None, "ADAPTIVE_LR_MIN": lo, "ADAPTIVE_LR_MAX": hi,
        "OPTIMIZER_TYPE": optimizer, "OPTIMIZER_ARGS": MUON_DEFAULT_ARGS if optimizer == "muon" else "",
        "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
        "DATASET_MEGAPIXELS": mp, "FAMILY_EMA": "0.98 (recommended)", "COMPILE_BLOCKS": "Auto",
        "BLOCKS_SWAP": "Auto (detect from GPU)", "FAMILY_PRECISION": "Auto (fits your free VRAM)",
        **({"FAMILY_FT_ROTATIONS": "3", "FAMILY_FT_ROTATE_EVERY": "2", "FAMILY_FT_SAVE_EVERY": "1"} if ft else {}),
    }


_FIBO2 = FamilyDescription(
    key="fibo2",
    arch_id="fibo2",
    display_name="Fibo 2.0",
    gui_label="Fibo 2.0",
    lora_name_suffix="fibo2",
    aliases=("fibo-2", "fibo2-comfyorg"),
    experimental=True,
    # build mode (docs/drivers/README.md): the family stays out of the GUI until every CHECKLIST gate has its
    # evidence, and un-hides in one deliberate commit once they pass (the close-out todo). CLI training and the
    # registry are unaffected by the flag - it is a GUI-visibility decision, not a disable.
    # flipped at the user's request (the training-run proofs that gate the un-hide are deferred):
    hidden=False,

    model_files=(
        ModelFile("fibo2_dit", "Fibo 2.0 DiT", True, "Comfy-Org/Fibo-2.0",
                  "diffusion_models/fibo2_dev_bf16.safetensors", 16.45,
                  "The ComfyUI single-file dev checkpoint - the build people already have for inference; the "
                  "family un-fuses it at load and re-fuses the LoRA on export.",
                  role="dit",
                  hint="Comfy-Org's single-file dev build (fused attention.qkv / ffn.w13, the FFN hidden "
                       "zero-padded to 9728): load_dit splits it back into the Linears the block map names and "
                       "trims the padding, so it trains the same model as the upstream diffusers shards "
                       "(briaai/Fibo-2.0-dev, transformer/), which load too. The save writes the fused layout "
                       "back out, so the LoRA loads in ComfyUI as saved.",
                  download_note="~16.5 GB - Comfy-Org/Fibo-2.0 -> diffusion_models/fibo2_dev_bf16.safetensors "
                                "(the fused single file; the sharded briaai repo reaches the same model)",
                  # measured 4 Nov 2026: loads all 783 tensors, a 768 render 5/255 (mean) from the bf16 file's
                  alt_repo="Comfy-Org/Fibo-2.0", alt_path="diffusion_models/fibo2_dev_int8_convrot.safetensors",
                  alt_label="int8",
                  alt_note="~8.4 GB int8 ConvRot, the file ComfyUI's Fibo 2.0 workflows use; decoded at load",
                  alt_size_gb=8.41),
        ModelFile("fibo2_vae", "Fibo 2.0 autoencoder", True, "Comfy-Org/Fibo-2.0", "vae/flux2-vae.safetensors", 0.34,
                  "The FLUX.2-family autoencoder Bria ships with Fibo 2.0 (the stock flux2-vae, diffusers keys).",
                  role="vae",
                  hint="32 latent channels at /8, normalized with the encoder's BatchNorm over the 2x2-packed "
                       "latents - the same autoencoder MOSS-Image and Klein carry, loaded through the shared "
                       "shape-built loader. The upstream repo's vae/ dir, the weights file itself, or a "
                       "parent-dir path all load.",
                  download_note="~0.34 GB - Comfy-Org/Fibo-2.0 -> vae/flux2-vae.safetensors (the stock FLUX.2 "
                                "autoencoder; any copy of it loads, whatever repo it is filed under)"),
        ModelFile("fibo2_text_encoder", "Fibo 2.0 text encoder", True, "Comfy-Org/Fibo-2.0",
                  "text_encoders/qwen3vl_4b_bf16.safetensors", 8.88, "The text encoder (Qwen3-VL 4B): loaded whole onto the target device, unloaded after caching.",
                  role="text_encoder", local_name="fibo2_text_encoder.safetensors",
                  hint="Qwen3-VL 4B Instruct - the same encoder Krea 2 and FLUX.3 use. Loaded whole onto the GPU "
                       "for encoding (the pipeline runs it with output_hidden_states=True and takes the selected "
                       "layers' hidden states), then offloaded before the step loop. The tokenizer/processor "
                       "comes from the copy bundled with Fizgig (offline); the Hub cache is the fallback.",
                  download_note="~8.9 GB - Comfy-Org/Fibo-2.0 -> text_encoders/qwen3vl_4b_bf16.safetensors (the "
                                "same bf16 build as briaai/Fibo-2.0-dev's, which is gated; fetched as "
                                "fibo2_text_encoder.safetensors)"),
        ModelFile("fibo2_turbo_lora", "Turbo LoRA (previews)", False, "Comfy-Org/Fibo-2.0",
                  "loras/fibo_2.0_turbo_lora_avg_rank_270_bf16.safetensors", 1.99,
                  "Comfy-Org's Turbo LoRA for the Dev model: previews and the workbench in 4 steps at CFG 1.",
                  role="speed_lora", fetch_optional=False),
    ),
    text_encoder_label="Fibo 2.0 text encoder",
    vae_label="Fibo 2.0 autoencoder",
    # The Qwen3-VL builds the driver knows: the base 4B Instruct, measured next to it.
    text_encoder_builds=(
        # measured 26 Oct 2026 (scripts/check_te_builds.py): the base build loads through fibo2's own loader and
        # conditions identically to the family's default file - cosine 1.00000, worst token 1.0000. Declared via
        # the Comfy-Org distribution because the upstream Qwen repo is sharded (no single file to resolve).
        TextEncoderBuild("Qwen3-VL 4B Instruct (base)", "Comfy-Org/Krea-2",
                         "text_encoders/qwen3vl_4b_bf16.safetensors", 8.88,
                         "36f3ff447ef59201722e8f9ce6020c9819fdcfba6aa2608c4e09b1c0ce114e34", 1.0, False,
                         "The base Qwen3-VL 4B Instruct in bf16 - conditions identically to the family's default "
                         "file (cosine 1.00000, worst token 1.0000)."),
    ),

    latent_channels=32,               # the FLUX.2 autoencoder's latents before its 2x2 pack, as the cache stores them
    spatial_factor=8,                 # /8 (the DiT patchifies 2x2 itself)
    bucket_step=16,
    image_channels=3,
    media=("photo",),                 # stills only: Fibo 2.0 trains single images, never clips or voice
    native_megapixels=1.0,

    n_blocks=44,                      # the 44 transformer blocks (layers.N); block_map adds the 2 refiner (noise_refiner.N) and 2 perceiver groups
    block_prefix="layers",
    block_note="44 transformer blocks (layers.N), 2 noise-refiner blocks (noise_refiner.N) and 2 perceiver layers.",

    lora=LoRAFormat(
        key_template="{prefix}.{module}.{ab}.weight",
        down="lora_down", up="lora_up",
        block_modules=_MODULES,
        file_prefix="",
        alpha_key="{prefix}.alpha",
        kohya=True,                     # the form ComfyUI's loader registers on this checkpoint
        # ComfyUI's Fibo 2.0 keeps attention.qkv, to_kv and w13 fused and maps no split keys onto them, so every
        # file is written in that layout (driver.export_lora_state_dict) and read back split
        save_fused=True,
        note="kohya keys (module path with dots as underscores), as every Fizgig Fibo 2.0 LoRA; ComfyUI reads them. "
             "The save writes the checkpoint's fused layout (attention.qkv, ffn.w13 with the FFN hidden padded "
             "to 9728) so the file loads in ComfyUI as saved; driver.convert_lora_state_dict maps a fused file "
             "back onto the split Linears the model trains against.",
        source="measured on the real Fibo 2.0 model in ComfyUI: an export writing the bare dotted paths resolved "
               "0/210 keys (each reported as 'lora key not loaded'), the kohya form resolved all 210",
    ),

    driver="fizgig.fibo2.driver:Fibo2Driver",
    modelspec_arch="BriaFibo2Transformer2DModel",
    implementation="briaai/Fibo-2.0-dev",
    # adamw first, as docs/drivers/CHECKLIST.md prescribes for a LoRA family: 8-bit AdamW steps each LoRA tensor
    # separately, and on a model with 415 adapted Linears that costs a large share of every step. The 8-bit
    # optimizer stays available behind it for the cards that need it.
    optimizers=("adamw", "adamw8bit", "automagic3"),
    ema_default="0.98",
    finetune=True,                    # driver.ft_spec: the 44 main layers on an NF4 trunk, in five parts
    ft_learning_rate=1e-5,            # the rate ticking Fine-tune sets (the guide's default until a run says otherwise)
    # train_memory: the per-block saving is the
    # checkpoint's own parameter bytes per main layer: the 44-block stack carries 7.43B of the model's 8.23B, so
    # 168.9M params a block - 0.314 GiB resident in bf16, 0.157 GiB in int8. NF4 stays 0.0: load_base refuses to
    # stream a 4-bit base, so nothing parks and nothing moves.
    # Re-measured 4 Nov 2026 on an RTX 5090 (eager, checkpointed, no swap, rank 16, a 1024 preview after each of 2
    # epochs): the preview sets every peak, so the figures are the preview's reserved peak (training's alone: bf16
    # 18.5 / 21.5, int8 10.6 / 14.1, nf4 8.3 / 11.2 at 0.25 / 1 MP). The 1 MP points trained on real photos above
    # 1 MP (buckets 1248x832, 1360x768). Swap check: bf16 with 26 blocks swapped peaked 10.6 GB against 18.5, 0.30 GB
    # a block, beside the 0.314 from the file. With the Turbo LoRA's 4-step previews (the default) the preview peaks
    # are bf16 26.6 / 27.8, int8 18.7 / 20.4, nf4 15.5 / 17.4 (it sits on the GPU while a preview renders); each
    # figure is the higher of the two, so it holds with the Turbo on or off.
    train_memory={"bf16": (((0.25, 26.6), (1.0, 27.8)), 0.314), "int8": (((0.25, 18.7), (1.0, 20.5)), 0.157),
                  "nf4": (((0.25, 15.5), (1.0, 17.5)), 0.0)},
    # under 20 GB the trainer parks the model for each preview's decode and renders at 768, so the 1024 preview's
    # peak above doesn't apply. Measured 6 Nov 2026 on a 16 GB RTX 4090 Laptop GPU (rank 32, automagic3, EMA, 16
    # photos at 0.25 MP, a 768 Turbo preview each epoch): INT8 with no swap peaked 12.4 GB reserved, 2.09 s a step
    # (the old plan, INT8 + 35 swapped blocks: 3.38 s). That is training's own 5090 peak (10.6) + 1.8 GB for the
    # preview; the other points carry the same +1.8 (5090 training-only: int8 14.1 at 1 MP, nf4 8.3 / 11.2).
    train_memory_small={"int8": (((0.25, 12.4), (1.0, 15.9)), 0.157), "nf4": (((0.25, 10.1), (1.0, 13.0)), 0.0)},
    # Big cards (40 GB and up), re-measured 4 Nov 2026 on an RTX PRO 6000 (96 GB): compiled, rank 32, AdamW, accum 1,
    # no swap, a 1024 preview after each of 2 epochs, 16 photos at 0.25 MP and 8 above 1 MP (buckets 1248x832,
    # 1360x768). Reserved peaks, previews included, with gradient checkpointing on / off:
    #   bf16 0.25 MP 22.4 / 30.6, 1 MP 24.5 / 55.8;  int8 0.25 MP 14.9 / 23.7, 1 MP 17.0 / 47.9
    # and with the Turbo LoRA's 4-step previews (the default):
    #   bf16 0.25 MP 24.2 / 30.4, 1 MP 24.2 / 55.7;  int8 0.25 MP 16.6 / 22.7, 1 MP 16.6 / 48.0
    # Each figure below is the higher of the two, so it holds with the Turbo on or off.
    # NF4 has no row: it keeps every dequantised bf16 weight for the backward without checkpointing, so like every
    # family's NF4 it always trains checkpointed.
    train_memory_nogc={"bf16": ((0.25, 30.6), (1.05, 55.8)), "int8": ((0.25, 23.7), (1.05, 48.0))},
    # (blocks, ((megapixels, GB with every block checkpointed, GB per block without it), ...)): the per-block cost is
    # the all-off slope from the same pair, (off - on) / 44 blocks, the steeper of Turbo off and on - bf16 0.185
    # (30.6 - 22.4) and 0.715 (55.7 - 24.2); int8 0.201 (23.7 - 14.9) and 0.713 (48.0 - 16.6) - and the all-on
    # figure the higher of the two (bf16 24.2 / 24.5, int8 16.6 / 17.0).
    train_memory_partial={"bf16": (44, ((0.25, 24.2, 0.185), (1.05, 24.5, 0.715))),
                          "int8": (44, ((0.25, 16.6, 0.201), (1.05, 17.0, 0.713)))},
    # compiles: the three block lists the driver compiles - layers (44), noise_refiner (2) and perceiver.layers
    # (2), so 48 blocks in total, exactly what F1 logged. There is no context_embedder on the diffusers model: the
    # perceiver *is* its context path, so no fourth list is missing here.
    compiles=True,
    compile_boundary="outside",       # as Flux 3's and MOSS-Image's: the checkpoint wraps the compiled block
    # comfy-kitchen's INT8 attention for renders (fibo2/model.py's attention() tries attend() first
    # and falls back when it returns None; training never turns the switch on). Measured on this card at the real
    # shape: a loss below ~1300 tokens (0.78x at 961, 0.86x at 1054), break-even ~1408, wins from 1664 (1.35x) up
    # to 1.73x at 16384 - so the driver gates the kernel at 2048, the first size where it is clearly and reliably
    # faster. Error per call 1.48-1.65% against bf16 SDPA's ~0.23% (the contributor's measurements, PR #250).
    int8_attention=True,
    # Compiled with the checkpoint outside the graph, re-measured 4 Nov 2026 on an RTX 5090 (rank 16, 2 epochs, a 1024
    # preview after each; the preview sets the peak). The 96 GB card's checkpointed runs above read 2-3 GB lower;
    # these are the higher. With the Turbo LoRA's previews: bf16 26.6 / 27.0, int8 18.7 / 19.7, nf4 15.5 / 15.9; each
    # figure is the higher of Turbo on and off. The inside boundary isn't measured on this code, so it has no row.
    compile_memory={"bf16": {"outside": ((0.25, 26.6), (1.05, 27.0))},
                    "int8": {"outside": ((0.25, 18.7), (1.05, 19.7))},
                    "nf4": {"outside": ((0.25, 15.5), (1.05, 15.9))}},
    # measured 25 Oct 2026, RTX PRO 6000 96GB (DDR4 host, warm inductor cache), bf16 DiT, 0.25 MP dataset
    # (113 images over 6 buckets), 5-min A/B at the "outside" boundary: --compile_blocks off 525 steps,
    # on 586 steps (0.571 -> 0.512 s/step, ~1.12x). Compiled overtakes eager by ~590 steps; 600 rounds it up.
    # A cold-cache first run pays a longer compile pause, so its crossover sits later than this figure.
    # 4 Nov 2026, RTX 5090, master's INT8 / NF4 code, 2 epochs, eager vs compiled at the same settings (epoch 2):
    # 0.25 MP bf16 0.77 -> 0.61, int8 0.82 -> 0.47, nf4 0.78 -> 0.70 s/step; 1 MP bf16 3.13 -> 2.49, int8 3.27 ->
    # 2.18, nf4 3.22 -> 2.58. On those 1-2 bucket sets the pauses paid back in ~195 (bf16), ~100 (int8) and ~470
    # (nf4, 0.25 MP) steps; each bucket shape pauses again, and bf16's 6-bucket 600 above is 3x its 195, so the
    # quantised figures take the same factor. (The 26 Oct finding that int8 / nf4 ran slower compiled was on the
    # INT8 storage this branch no longer changes.)
    compile_payback_steps={"bf16": 600, "int8": 300, "nf4": 1400},
    # LoRA only: a split-layout Kronecker adapter cannot be re-fused onto this checkpoint's packed qkv/w13/to_kv
    # tensors (each part trains its own w1), so LoKR has no layout that loads in the fused build and is not offered.
    network_types=("lora",),
    masked_training=True,             # training_loss takes loss_mask (and the prior outside it)
    batch_training=True,              # Batch size (Other Options): training_loss_batch, exact per picture
    batch_gb=(0.9, 1.7),            # GB each extra picture takes: fixed, per MP (batch 1 vs 3)
    # Batch x checkpointing sweep, 6 Nov 2026 (RTX PRO 6000, 8 pictures, 2 epochs, uncompiled, previews off - so the
    # no-checkpointing peaks below are the larger of that run's and train_memory's preview peak; uncompiled sits
    # above a compiled run, the safe side). Desktop\Fizgig batch tests\bsgc\MORNING_REPORT.md
    batch_auto=((0.25, 2), (1.0, 1)),          # 1.7x a picture at 0.25 MP; batch 2 at 1 MP didn't fit 96 GB
    batch_gb_nogc=((0.25, 22.0),),
    slider_training=True,             # photo pairs (training_loss diff weighting) or prompts (noise_latents / predict)
    # edit pairs as the pipeline edits: Qwen3-VL sees each before-image (encode_text_with_references) and the DiT
    # takes them as clean context latents (training_loss / generate refs)
    edit_training=True,
    edit_references=10,
    edit_note=("Pairs of an original and its edited version, the same crop and shape. The caption is the edit instruction ('make it night'), or a JSON caption of the result with an edit_instruction key."),
    activation_cache=True,            # Reuse unchanged blocks: step-1 replay over layers / refiners / Perceiver layers
    # int8/nf4: offered with measured peaks and the attention measurement.
    precisions=("int8", "nf4", "bf16"),
    # Auto plans from the measured figures.
    auto_precisions=("int8", "nf4"),

    sampling=(
        # The reference is diffusers' own BriaFibo2Pipeline (v0.42.1): 30 steps at guidance_scale 5.0, its negative
        # an empty structured caption when none is given. The model is not CFG-distilled, so CFG > 1 is the intended
        # use and the negative prompt does fire; 20 steps was too low for Dev. The card's 50-step figure is the
        # same recipe with more steps, not a different one.
        SamplingSettings("Fibo 2.0", steps=30, cfg=5.0, sampler="euler", scheduler="simple",
                         negative_prompt=True, note="Flow matching on the reference schedule; CFG 5 against the "
                         "negative prompt, as the diffusers pipeline defaults to it.",
                         source="diffusers 0.42.1 BriaFibo2Pipeline.__call__ defaults (num_inference_steps=30, "
                                "guidance_scale=5.0, DEFAULT_NEGATIVE_PROMPT)"),
    ),
    # Comfy-Org's Turbo LoRA (the Dev-to-Edit-Turbo difference, dynamic rank, sv_fro 0.95): 4 steps at CFG 1, Euler
    # on the simple scheduler, as Comfy-Org's Fibo 2.0 Turbo workflow samples. Its low-rank pairs cover all 415
    # Linears (the fused qkv / to_kv / w13 split exactly by convert_lora_state_dict) and its bias deltas load; its
    # norm-weight deltas (.diff) are not applied, as on Qwen Image 2.1's Turbo LoRA.
    speed_loras=(
        SpeedLoRA(
            name="Turbo LoRA (4-step)",
            repo="Comfy-Org/Fibo-2.0",
            file="loras/fibo_2.0_turbo_lora_avg_rank_270_bf16.safetensors",
            pairs_with="Fibo 2.0",
            strength=1.0,
            settings=SamplingSettings("Turbo LoRA 4-step", steps=4, cfg=1.0, sampler="euler", scheduler="simple",
                                      note="Comfy-Org's Fibo 2.0 Turbo workflow: 4 steps, CFG 1.",
                                      source="https://github.com/Comfy-Org/workflow_templates/blob/main/templates/"
                                             "image_fibo2_edit_turbo_pe.json"),
            load_unmerged=True,
            pref_key="fibo2_turbo_lora",
            source="https://huggingface.co/Comfy-Org/Fibo-2.0",
        ),
    ),
    preview_speed_lora="Turbo LoRA (4-step)",
    preview_speed_steps=4,            # on by default, as Qwen Image 2.1's Turbo LoRA: 4-step previews
    preview_speed_strength=1.0,
    preview_steps=30,
    preview_cfg=5.0,
    # the pipeline's own negative when none is given (diffusers 0.42.1 DEFAULT_NEGATIVE_PROMPT): an empty structured
    # caption, not an empty string
    preview_negative=('{"short_description":"","objects":[],"background_setting":"","lighting":{},"aesthetics":{},'
                      '"photographic_characteristics":{},"style_medium":"","context":"","artistic_style":""}'),
    workbench_follows_samples=True,   # CFG 5: the workbench takes the Samples tab's negative, steps and CFG
    # a third to three times the AdamW rate measured to train cleanly (1e-4, 4 Nov 2026); Automagic sets its own rate,
    # so the built-in presets never warn
    start_lr_bands=(("lora", 3e-5, 3e-4), ("finetune", 3.3e-6, 3e-5), ("slider", 7e-5, 6e-4)),  # from the presets
    preview_width=1024,
    preview_height=1024,

    # workbench: the tabs the family can render, on the block map's ids.
    workbench=("repair", "profiler", "extract", "royale"),
    # Repair Studio at 1024 for the 1 MP model (the 1.0 native_megapixels).
    repair_size=1024,
    samples_text=(),  # the family's sampler is in sampling (Euler, simple scheduler, flow matching)
    # The built-in Training-tab presets: the first is applied on a first visit.
    presets=(
        ("✨ Fibo 2.0 (rank 32, full model)", _preset(32, lr=1e-4)),   # Muon (Peter, 6 Nov): a flat 1e-4, the rate AdamW was measured at
        # sliders are rank 8 on every model, AdamW at 2e-4 for 20 epochs at 0.5 MP as Flux 3's
        # edit pairs, as trained on 4 Nov 2026 (15 greyscale -> colour pairs, 6 epochs at 0.25 MP): AdamW at 1e-4
        ("✨ Fibo 2.0 Edit (rank 16)", _preset(16, lr=1e-4, epochs=6, edit=True)),
        # a full fine-tune, as run on 4 Nov 2026 (16 photos at 0.25 MP: 3 rotations, 2 epochs a part, its checkpoint
        # loaded in ComfyUI in place of the base and showed the training)
        ("✨ Fibo 2.0 Fine-tune (1e-5)", _preset(16, lr=1e-5, ft=True)),
        ("✨ Fibo 2.0 Slider (rank 8, 2e-4)",
         _preset(8, lr=2e-4, epochs=20, mp="0.5", slider=True)),
    ),
    # no Model Area to Train or Fast Identity Mode: those come from a measured block map (Klein's areas, Qwen Image's
    # identity blocks), which Fibo doesn't have. The LoRA trains the whole model, as Bria's reference trainer does
    identity_blocks=(),
    # The Extract tab's presets, on the block map's ids.
    extract_presets=(),
    # The Repair Studio presets, on the block map's ids.
    repair_presets=(),
    # LoRA Royale prompt travel needs pad_conditioning.
    notes=(
        ("The model trains on split Linears (diffusers' layout, fibo2/model.py); every LoRA file is written in "
         "ComfyUI's fused layout (LoRAFormat.save_fused, driver.export_lora_state_dict) and read back split "
         "(driver.convert_lora_state_dict).", "fibo2/driver.py"),
    ),
)

FIBO2 = _FIBO2
