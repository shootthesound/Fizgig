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

from fizgig.families.description import (
    FamilyDescription,
    LoRAFormat,
    ModelFile,
    SamplingSettings,
    TextEncoderBuild,
)

_MODULES = ("to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3")   # every Linear the LoRA targets

# The block groups the "Model Area to Train" presets are built from - every id resolves against the driver's
# block map (51 in total: 44 main blocks, 2 noise refiners, 2 context perceivers, 3 I/O). The I/O trio is
# pinned into all four presets, so no area leaves the model without its input, timestep and output layers.
_LAYERS = tuple(f"layers.{i}" for i in range(44))
_REFINERS = ("noise_refiner.0", "noise_refiner.1")
_PERCEIVERS = ("perceiver.layers.0", "perceiver.layers.1")
_IO = ("x_embedder", "t_embedder", "final_layer")

# the note both built-in presets carry beside the picker: the sampler is the reference pipeline's recipe, and it is
# the same whatever area is trained - only the rank and the block set move between them
_SAMPLING_NOTE = "Flow matching; CFG 5 against the negative prompt, as the reference pipeline."


def _preset(rank, area, lr=1e-4, adaptive=None, epochs=60, mp="0.25"):
    """The built-in presets' values, in the standard layer's keys (the shape krea2.py and moss_image.py use).
    adaptive=(min, max) turns Adaptive LR on; area is one of train_areas' names."""
    lo, hi = adaptive or ("1e-5", "4e-4")
    return {
        "NETWORK_DIM": rank, "NETWORK_ALPHA": rank, "NETWORK_TYPE": "LoRA (standard)",
        "TARGET_LAYERS": area, "FAMILY_EDIT": False, "FAMILY_SLIDER": False, "FAMILY_FT": False,
        "LEARNING_RATE": lr, "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": adaptive is not None, "ADAPTIVE_LR_MIN": lo, "ADAPTIVE_LR_MAX": hi,
        "OPTIMIZER_TYPE": "automagic3", "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
        "DATASET_MEGAPIXELS": mp, "FAMILY_EMA": "0.98 (recommended)", "COMPILE_BLOCKS": "Auto",
        "BLOCKS_SWAP": "Auto (detect from GPU)", "FAMILY_PRECISION": "Auto (fits your free VRAM)",
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
                                "(the fused single file; the sharded briaai repo reaches the same model)"),
        ModelFile("fibo2_vae", "Fibo 2.0 autoencoder", True, "Comfy-Org/Fibo-2.0", "vae/flux2-vae.safetensors", 0.34,
                  "The FLUX.2-family autoencoder Bria ships with Fibo 2.0 (the stock flux2-vae, diffusers keys).",
                  role="vae",
                  hint="32 latent channels at /8, normalized with the encoder's BatchNorm over the 2x2-packed "
                       "latents - the same autoencoder MOSS-Image and Klein carry, loaded through the shared "
                       "shape-built loader. The upstream repo's vae/ dir, the weights file itself, or a "
                       "parent-dir path all load.",
                  download_note="~0.34 GB - Comfy-Org/Fibo-2.0 -> vae/flux2-vae.safetensors (the stock FLUX.2 "
                                "autoencoder; any copy of it loads, whatever repo it is filed under)"),
        ModelFile("fibo2_text_encoder", "Fibo 2.0 text encoder", True, _CANON, "text_encoder/model.safetensors",
                  8.88, "The text encoder (Qwen3-VL 4B): loaded whole onto the target device, unloaded after caching.",
                  role="text_encoder", local_name="fibo2_text_encoder.safetensors",
                  hint="Qwen3-VL 4B Instruct - the same encoder Krea 2 and FLUX.3 use. Loaded whole onto the GPU "
                       "for encoding (the pipeline runs it with output_hidden_states=True and takes the selected "
                       "layers' hidden states), then offloaded before the step loop. The tokenizer/processor "
                       "comes from the copy bundled with Fizgig (offline); the Hub cache is the fallback.",
                  download_note="~8.9 GB - briaai/Fibo-2.0-dev -> text_encoder/model.safetensors (bf16; fetched "
                                "as fibo2_text_encoder.safetensors)"),
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

    latent_channels=128,              # Fibo 2.0 autoencoder latents
    spatial_factor=16,                # latents at /16
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
        # single output (docs/drivers/STILLS.md): the training save itself writes the fused+padded ComfyUI
        # layout via driver.export_lora_state_dict - no separate export step. The live model trains on the
        # split Linears; driver.convert_lora_state_dict is the read-side inverse that turns a fused file back
        # into them (resume states stay split: _save_state saves without fuse).
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
    # CC BY-NC 4.0, straight from the snapshot's own front-matter (license_name "bria-fibo", the deed at
    # https://creativecommons.org/licenses/by-nc/4.0/deed.en) and BRIA's README (commercial use needs a separate
    # agreement, https://platform.bria.ai/licensing/pricing?model=fibo-2). A LoRA trained on the base is a derivative
    # of it, so the restriction is stated where the model is picked, not in a footer. The card says nothing about
    # *outputs*, so this note claims nothing about them either.
    model_note="Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0): BRIA releases the Fibo 2.0 weights "
               "for non-commercial use only - https://creativecommons.org/licenses/by-nc/4.0/deed.en. Commercial "
               "use is subject to a separate agreement with BRIA - "
               "https://platform.bria.ai/licensing/pricing?model=fibo-2. A LoRA trained on this base is a "
               "derivative of it and carries the same non-commercial restriction.",
    # adamw first, as docs/drivers/CHECKLIST.md prescribes for a LoRA family: 8-bit AdamW steps each LoRA tensor
    # separately, and on a model with 415 adapted Linears that costs a large share of every step. The 8-bit
    # optimizer stays available behind it for the cards that need it.
    optimizers=("adamw", "adamw8bit", "automagic3"),
    ema_default="0.98",
    finetune=False,
    # train_memory: the trainer's logged reserved peaks, previews on (1024 canvas), 2 epochs, RTX PRO 6000 96GB,
    # gradient accumulation 32, checkpointing on, eager. The 0.25 MP points (496x496, 57 images) came from
    # desktop-app runs; the 1.05 MP points (1024x1024, same 57 images) from CLI runs A4 (bf16 22.1), A5 (int8 15.9)
    # and A6 (nf4 12.3 - the run's high-water, which is the max across epochs: the trainer resets the peak counter at
    # every epoch start, and A6's first epoch reached 12.3 where its last reported 12.2). The per-block saving is the
    # checkpoint's own parameter bytes per main layer: the 44-block stack carries 7.43B of the model's 8.23B, so
    # 168.9M params a block - 0.314 GiB resident in bf16, 0.157 GiB in int8. NF4 stays 0.0: load_base refuses to
    # stream a 4-bit base, so nothing parks and nothing moves.
    train_memory={"bf16": (((0.25, 20.0), (1.05, 22.1)), 0.314), "int8": (((0.25, 13.1), (1.05, 15.9)), 0.157),
                  "nf4": (((0.25, 10.2), (1.05, 12.3)), 0.0)},
    # measured: bf16 LoRA rank 32 / alpha 32, automagic3, lr 1e-4 constant, EMA 0.98, no swap,
    # previews on (1024 canvas), compiled at the family's "outside" boundary, gradient accumulation 1 (the only setting
    # in which the trainer consults the checkpointing planner at all), 2 epochs on the 57-image 496x496 dataset,
    # RTX PRO 6000 96GB: peak reserved 30.6 GB at 0.25 MP. The planner turns checkpointing off when this fits free
    # VRAM with NOGC_MARGIN_GB (4.0) to spare, so a 34.5 GB window stops authorizing it the moment the figure is real.
    # The 1 MP row needs a ~50 GB window; int8 and nf4 cannot be measured here until the int8 compile crash is fixed.
    # measured peaks (compiled, accum 1, previews on, rank 32, no swap), each replicated across both epochs: 30.6 GB at
    # 0.25 MP (both epochs identical) and 57.6 GB at 1.0 MP (both epochs identical). The checkpointing delta
    # over the compiled checkpointed baseline is 12.5 GB at 961 tokens but 36.9 GB at 4096 tokens — only 2.95x for 4.26x
    # the pixels — so the delta is sub-linear in tokens and cannot be projected from a single resolution.
    # Both quantised rows are MEASURED (26 Oct 2026), each with the planner answering off on a free ~92 GB card.
    # int8: 22.8 GB in both epochs, high-water 22.84 GB with the preview. Its 23.4 GB seed was derived as the
    # checkpointed peak (10.9) plus bf16's checkpointing delta (12.5), and the real int8 delta measured 12.2 GB -
    # within 0.3 GB, because checkpointing stashes ACTIVATIONS, which are bf16 whichever way the weights are stored.
    # nf4: 33.8 GB in both epochs, high-water 33.83 GB, against a 22.4 GB seed - wrong by 11.4 GB, and the reason
    # is not activations. nf4_linear_forward_patch dequantises the packed 4-bit weight to x.dtype and calls
    # F.linear, so autograd SAVES THAT BF16 WEIGHT for the input-gradient of each of the 415 quantised Linears;
    # with checkpointing off they are all resident at once, which is most of the bf16 weight set, so 4-bit storage
    # buys nothing on the memory axis. With checkpointing on only one block's copy is live (recomputed inside the
    # checkpoint), so the resident weights stay 4-bit. int8 pays neither: its matmul runs in 8-bit and no bf16
    # copy is ever materialised. The lesson for the next seed: a derived figure can only bootstrap the planner
    # into answering at all - never read it as a fit decision, because a 30 GB-free card would have been told nf4
    # fits uncheckpointed at 26.4 GB and would have OOMed on the real 37.8 GB.
    # THE FIX THAT DID NOT MOVE IT, same day: nf4 now goes through _NF4DequantLinear (packed weight + quant state
    # on the context, the bf16 copy rebuilt in the backward - bit-identical parity, an eager-mode residency win).
    # But compiling gives the residency straight back: inductor keeps the dequantised weights alive across the step
    # however the Python-level forward is written, so 33.8 GB stands unchanged for compiled nf4. The conversion
    # stays (exact, and it wins in eager); reclaiming the compiled gap needs bnb's C++ matmul_4bit, not a Python
    # autograd.Function. Full A/B numbers in the run sheet.
    train_memory_nogc={"bf16": ((0.25, 30.6), (1.05, 57.6)), "int8": ((0.25, 22.8),), "nf4": ((0.25, 33.8),)},
    # (blocks, ((megapixels, GB with every block checkpointed, GB per block without it), ...)).
    # The shipped pool is 44 — every block is available for un-checkpointing on a family with no swap. The house
    # protocol (as krea2.py:151-153 and qwen_image.py:160-162 state it): per_block = the LARGER of the half-off and
    # all-off slopes, both derived from measured endpoints. Applied here to fibo2's own measured figures:
    #   0.25 MP: all-on 18.1 (D2c2), half-off 22 blocks -> 18.6 (D2e) = 0.023/block; all-off 30.6 (nogc) = 0.284/block
    #           -> larger 0.284
    #   1.05 MP: all-on 20.7 (C1/B3), 8-off -> 20.7 (D2d/D2d2, identical twice) = ~0; all-off 57.6 (nogc) = 0.839/block
    #           -> larger 0.839
    # The 1.05 MP point previously carried 1.210, borrowed from B3's 0.25 MP delta and never measured at 1 MP - it now
    # carries the protocol figure from fibo2's own measured all-off slope. Every endpoint is a measurement; the
    # per_block figures are protocol-derived from them, exactly as the other families' rows are.
    # Note the empirical peak is flatter than the planner's linear model (half-off cost 0.5 GB where 0.284/block
    # predicts 6.2; 8-off at 1 MP cost 0.0 GB where 0.839/block predicts 6.7) - the larger-slope protocol is
    # deliberately conservative, so the planner un-checkpoints fewer blocks than it could. Never more.
    train_memory_partial={"bf16": (44, ((0.25, 18.1, 0.284), (1.05, 20.7, 0.839))),},
    # int8 + torch.compile: FIXED 26 Oct 2026 - a compiled INT8 run died at step 1, because the int8 bytes sat in
    # Parameter.data and FakeTensor cannot reconcile a Parameter whose dtype disagrees with the compute dtype. The
    # fix (NF4's shape: bytes in a non-persistent buffer, .weight a bf16 stub) is in modules/int8_train.py's
    # apply_int8_training, mirrored by families/quant.py's _quantize_one. Proof: run F1, 57/57 steps compiled,
    # 10.9 GB training peak, adapter saved.
    # compiles: the three block lists the driver compiles - layers (44), noise_refiner (2) and perceiver.layers
    # (2), so 48 blocks in total, exactly what F1 logged. There is no context_embedder on the diffusers model: the
    # perceiver *is* its context path, so no fourth list is missing here.
    compiles=True,
    compile_boundary="outside",       # as Flux 3's and MOSS-Image's: the checkpoint wraps the compiled block
    # comfy-kitchen's INT8 attention for renders (the driver's dispatch_attention_fn wrapper tries attend() first
    # and falls back when it returns None; training never turns the switch on). Measured on this card at the real
    # shape: a loss below ~1300 tokens (0.78x at 961, 0.86x at 1054), break-even ~1408, wins from 1664 (1.35x) up
    # to 1.73x at 16384 - so the driver gates the kernel at 2048, the first size where it is clearly and reliably
    # faster. Error per call 1.48-1.65% against bf16 SDPA's ~0.23%; the full table is in the run sheet.
    int8_attention=True,
    # measured 25-26 Oct 2026, CLI runs, compiled + checkpointed, previews on (1024 canvas), 2 epochs, 96GB card.
    # compile_plan reads these against free VRAM: the declared boundary ("outside") if it fits, else the other,
    # else uncompiled. outside = C1 (bf16 20.7) / C5r (nf4 10.2) / F1,H1 (int8 13.06, 13.55); inside = C2 (bf16
    # 21.4) - inside costs ~0.7 GB over outside at the same size. nf4's inside run was never taken, so nf4 has no
    # inside row. H1/H2 closed the int8-1MP and nf4-0.25MP gaps (accum 1, 1 epoch): at 0.25 MP the 1024 preview
    # owns the peak (nf4 9.68 high-water on a 7.6 compiled training peak), at 1 MP the training pass does (int8
    # 13.55 on a 13.0 peak, +0.5 GB for 4x the pixels). Rows taken at accum 32 vs 1 differ by under 0.4 GB on the
    # peaks, so the rows compare without a re-take.
    compile_memory={"bf16": {"outside": ((1.05, 20.7),), "inside": ((1.05, 21.4),)},
                    "nf4": {"outside": ((0.25, 9.68), (1.05, 10.2)),},
                    "int8": {"outside": ((0.25, 13.06), (1.05, 13.55)),}},
    # measured 25 Oct 2026, RTX PRO 6000 96GB (DDR4 host, warm inductor cache), bf16 DiT, 0.25 MP dataset
    # (113 images over 6 buckets), 5-min A/B at the "outside" boundary: --compile_blocks off 525 steps,
    # on 586 steps (0.571 -> 0.512 s/step, ~1.12x). Compiled overtakes eager by ~590 steps; 600 rounds it up.
    # A cold-cache first run pays a longer compile pause, so its crossover sits later than this figure.
    # 26 Oct 2026, eager (P1, P2) against compiled (F1, H2), free card, 0.25 MP, accum 1, checkpointed: int8 1.03
    # s/step eager vs 1.36 compiled steady (25.3 s pause); nf4 0.91 vs 1.10 (19.5 s). BOTH quantised bases are
    # slower compiled and pay the pause up front, so the crossover never arrives - None, not a step count. bf16 is
    # the opposite (0.88 -> 0.71, 600 steps). Hypothesized mechanism (rates and pauses measured, mechanism not
    # proven): inductor unpacks the quantised weights to full precision inside the compiled graph, losing the
    # kernel advantage. Compiling buys them ~0.8 GB each (int8 11.7 -> 10.9, nf4 8.4 -> 7.6 training peak), so
    # their trade is speed against memory. With None here Auto declines to compile either base on any card;
    # --compile_blocks on forces it at the measured cost.
    compile_payback_steps={"bf16": 600, "int8": None, "nf4": None},
    # LoRA only: a split-layout Kronecker adapter cannot be re-fused onto this checkpoint's packed qkv/w13/to_kv
    # tensors (each part trains its own w1), so LoKR has no layout that loads in the fused build and is not offered.
    network_types=("lora",),
    # int8/nf4: offered with measured peaks and the attention measurement.
    precisions=("int8", "nf4", "bf16"),
    # Auto plans from the measured figures.
    auto_precisions=("int8", "nf4"),

    sampling=(
        # The reference is diffusers' own BriaFibo2Pipeline (verified 26 Oct 2026): 30 steps at guidance_scale 5.0,
        # negative_prompt optional (empty by default). The model is not CFG-distilled, so CFG > 1 is the intended
        # use and the negative prompt does fire; 20 steps was too low for Dev. The card's 50-step figure is the
        # same recipe with more steps, not a different one.
        SamplingSettings("Fibo 2.0 Dev", steps=30, cfg=5.0, sampler="euler", scheduler="simple",
                         negative_prompt=True, note="Flow matching on the reference schedule; CFG 5 against the "
                         "negative prompt, as the diffusers pipeline defaults to it.",
                         source="diffusers 0.42.0.dev0 BriaFibo2Pipeline.__call__ defaults (num_inference_steps=30, "
                                "guidance_scale=5.0, negative_prompt optional)"),
    ),
    preview_steps=30,
    preview_cfg=5.0,
    preview_negative=None,
    preview_width=1024,
    preview_height=1024,

    # workbench: the tabs the family can render, on the block map's ids.
    workbench=("repair", "profiler", "extract", "royale"),
    # Repair Studio at 1024 for the 1 MP model (the 1.0 native_megapixels).
    repair_size=1024,
    samples_text=(),  # the family's sampler is in sampling (Euler, simple scheduler, flow matching)
    # The built-in Training-tab presets: the first is applied on a first visit. Full Model at rank 32 is the general
    # default; the rank-16 area leaves the refiners and the perceivers out, the shape the 0.25 MP probes trained.
    presets=(
        ("✨ Fibo 2.0 Dev (rank 32, full model)", _preset(32, "Full Model")),
        ("✨ Fibo 2.0 Dev (rank 16, skip refiner + perceiver blocks)",
         _preset(16, "Full without refiners or perceivers")),
    ),
    # notes beside Load Preset: (preset name, note) pairs
    preset_notes=(("✨ Fibo 2.0 Dev (rank 32, full model)", _SAMPLING_NOTE),
                  ("✨ Fibo 2.0 Dev (rank 16, skip refiner + perceiver blocks)", _SAMPLING_NOTE)),
    # The Model Area to Train presets: (name, (block ids), (min_t, max_t) or None). The first is the default; the
    # window is None on all four, so the Timestep Range control stays unused. The reference trains all 415
    # nn.Linear layers - that is "Full Model", the 51-block row.
    train_areas=(
        ("Full Model", _LAYERS + _REFINERS + _PERCEIVERS + _IO, None),          # 51 blocks
        ("Full without refiners", _LAYERS + _PERCEIVERS + _IO, None),           # 49
        ("Full without perceivers", _LAYERS + _REFINERS + _IO, None),           # 49
        ("Full without refiners or perceivers", _LAYERS + _IO, None),          # 47
    ),
    # The four areas already partition the whole model, so a tick list over the same blocks adds nothing: Custom
    # is not offered. A saved run that still names "Custom" resolves through families/launch.py's area_blocks.
    train_area_custom=False,
    identity_blocks=(),
    # The Extract tab's presets, on the block map's ids.
    extract_presets=(),
    # The Repair Studio presets, on the block map's ids.
    repair_presets=(),
    # LoRA Royale prompt travel needs pad_conditioning.
    notes=(
        ("The base is the upstream diffusers repo (diffusers key layout); the training save is the split LoRA, "
         "and a ComfyUI-fused export is an explicit transform (driver.export_lora_state_dict).", "fibo2/driver.py"),
    ),
)

FIBO2 = _FIBO2
