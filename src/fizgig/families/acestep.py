"""ACE-Step 1.5 XL through the standard layer: text-to-music (songs with vocals and lyrics, instrumentals), 48 kHz
stereo. A 4B flow DiT (32 layers, 2560 wide) over an Oobleck VAE, Qwen3-Embedding-0.6B text and lyric conditioning.
Driver: acestep/driver.py.

Facts from the official repository (github.com/ace-step/ACE-Step-1.5: the handler, trainer and docs; the XL model
repos' code), Side-Step's guides and ComfyUI's implementation (comfy/ldm/ace/ace_step15.py, text_encoders/ace15.py,
lora.py), Oct 2026.
"""
from fizgig.families.description import FamilyDescription, FamilyOption, LoRAFormat, ModelFile, SamplingSettings

_COMFY = "Comfy-Org/ace_step_1.5_ComfyUI_files"
_REPO = "https://github.com/ace-step/ACE-Step-1.5"
_TURBO_T = (1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3)


def _preset(rank, alpha, lr, epochs, kind="LoRA (standard)", ema="0.98 (recommended)", save_every=40):
    return {
        "NETWORK_DIM": rank, "NETWORK_ALPHA": alpha, "NETWORK_TYPE": kind, "LEARNING_RATE": lr,
        "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": save_every, "SEED": 42, "ADAPTIVE_LR": False,
        "OPTIMIZER_TYPE": "adamw", "GRADIENT_ACCUMULATION": 4, "MAX_GRAD_NORM": 1.0,
        "BLOCKS_SWAP": "Auto (detect from GPU)", "FAMILY_PRECISION": "Auto (fits your free VRAM)",
        "FAMILY_EMA": ema, "KREA2_LOSS_WATCH": False, "KREA2_PER_IMAGE_LR": False, "KREA2_AUTO_RECAPTION": False,
        "KREA2_WARMUP_LOOK": False, "FAMILY_FT": False, "FAMILY_SLIDER": False,
    }


ACESTEP15 = FamilyDescription(
    key="acestep15xl",
    caption_dropout=False,                   # songs: no captions to drop
    resolution_stages=False,                 # songs: no picture size
    detail_crops=False,                      # songs: no pictures to crop
    start_lr_bands=(("lora", 3e-5, 3e-4), ("finetune", 3e-6, 3e-5), ("slider", 7e-5, 6e-4)),
    arch_id="acestep15xl",
    display_name="ACE-Step 1.5 XL",
    gui_label="ACE-Step 1.5 XL (music)",
    lora_name_suffix="acestep15xl",
    aliases=("ace-step", "acestep", "ace-step-1.5-xl", "acestep15"),
    experimental=True,

    model_files=(
        ModelFile("acestep15xl_dit", "ACE-Step 1.5 XL Turbo DiT", True, _COMFY,
                  "split_files/diffusion_models/acestep_v1.5_xl_turbo_bf16.safetensors", 9.97, role="dit",
                  hint="The XL Turbo model (8 steps): what LoRAs train on by default, and what the workbench and "
                       "previews render with."),
        ModelFile("acestep15xl_sft_dit", "ACE-Step 1.5 XL SFT DiT", False, _COMFY,
                  "split_files/diffusion_models/acestep_v1.5_xl_sft_bf16.safetensors", 9.97,
                  hint="Optional: the non-distilled XL model, for the SFT Training Base (30-step previews with "
                       "guidance - slower, the most detail)."),
        ModelFile("acestep15xl_vae", "ACE-Step 1.5 VAE", True, _COMFY, "split_files/vae/ace_1.5_vae.safetensors", 0.34,
                  role="vae", hint="The Oobleck audio VAE: 48 kHz stereo to 64 channels at 25 per second."),
        ModelFile("acestep15xl_model_eq", "Model EQ LoRA (Fizgig)", False, "ShootTheSound/Fizgig-ACE-Step-1.5-XL-Model-EQ",
                  "acestep15xl_turbo_model_eq_r128.safetensors", 0.34, role="model_eq", fetch_optional=True,
                  hint="Optional: ACE-Step XL Turbo's own weights as a LoRA. In Repair Studio its block sliders "
                       "become an EQ of the model itself - 1.0 is the stock model - and a save bakes your EQ into "
                       "an ordinary LoRA. Load it as the primary or the donor with the Model EQ buttons."),
        ModelFile("acestep15xl_text_encoder", "Qwen3 0.6B text encoder", True, _COMFY,
                  "split_files/text_encoders/qwen_0.6b_ace15.safetensors", 1.19, role="text_encoder",
                  hint="Encodes the captions and lyrics for caching, then unloads before training."),
    ),
    prefs_title="Model Paths (ACE-Step 1.5 XL)",
    text_encoder_label="Qwen3 0.6B",
    vae_label="ACE-Step 1.5 VAE",

    media=("voice",),
    subject_isolation=False,         # songs: no DOP and no LoRA Isolation (the Subjects card keeps its subject folders)
    latent_channels=64,
    spatial_factor=1,
    bucket_step=1,
    native_megapixels=1.0,

    n_blocks=32,                      # config.json num_hidden_layers
    block_prefix="layers",
    block_note="32 DiT layers, sliding-window (even) and full (odd) self-attention, each with cross-attention to "
               "the caption, lyrics and timbre. The LoRA wraps q/k/v/o of both attentions (the official trainer's "
               "targets).",
    workbench=("repair", "extract"),
    # I'm feeling lucky's held bands, from Peter's by-ear block map (10 Oct 2026, fzgfunk): 0-5 structure (groove,
    # arrangement) and 28-31 the tidy-up of detail stay near 1 while Hold is ticked; 6-27 roll the full Min-Max range
    lucky_zones=((0, 5, 0.8, 1.2), (28, 31, 0.9, 1.1)),
    # Peter's Repair Studio mixes (Oct 2026, by ear); every layer not listed stays at 1.0
    repair_presets=(
        ("✨Demo Slider Adjustment",
         (("layer_6", -1.4516), ("layer_7", 0.1935), ("layer_8", 0.5806), ("layer_9", 0.6774),
          ("layer_10", 0.7742), ("layer_11", 0.7742), ("layer_12", 0.2903), ("layer_13", 0), ("layer_14", 0),
          ("layer_15", 0), ("layer_16", 0), ("layer_17", 0), ("layer_18", 0), ("layer_19", 0),
          ("layer_20", 0), ("layer_21", 0), ("layer_22", 0), ("layer_23", 0), ("layer_24", 0),
          ("layer_25", 0), ("layer_26", 0), ("layer_27", 0))),
        ("✨Early Epoch Improver",
         (("layer_0", 0.3871), ("layer_1", 0.5806), ("layer_2", 0.5806), ("layer_3", 0.6774),
          ("layer_5", 0.7742), ("layer_6", -0.15), ("layer_8", -0.1), ("layer_9", 0.15),
          ("layer_10", -0.9677), ("layer_11", -0.95), ("layer_12", 0.4), ("layer_13", -0.871),
          ("layer_14", 0.85), ("layer_15", -0.6774), ("layer_16", 0.95), ("layer_17", -1.3),
          ("layer_18", 0.6), ("layer_19", 0), ("layer_20", -1.6452), ("layer_21", 0.6), ("layer_22", 1.3),
          ("layer_23", -0.55), ("layer_24", -1.5484), ("layer_25", 1.05), ("layer_26", 0.35),
          ("layer_27", 0.95), ("layer_28", 1.05), ("layer_29", 1.05), ("layer_30", 0.95), ("layer_31", 0.9))),
        ("✨Make it more full",
         (("layer_0", 0.1935), ("layer_1", 0.2), ("layer_2", 0.1935), ("layer_3", 0.1935),
          ("layer_4", 0.1935), ("layer_5", 0.1935), ("layer_6", 0.1935), ("layer_7", 0.1935),
          ("layer_8", -1.3548), ("layer_9", -0.85), ("layer_10", -1.0645), ("layer_11", 1.35),
          ("layer_12", 0.55), ("layer_13", -1.2581), ("layer_14", 0.871), ("layer_15", -0.871),
          ("layer_16", 0.4), ("layer_18", -2.0323), ("layer_19", -1.4), ("layer_20", -2.7),
          ("layer_21", -1.65), ("layer_22", -1.2), ("layer_23", -1.7419), ("layer_24", 0.5),
          ("layer_25", -2.129), ("layer_26", -1.65), ("layer_27", -1.5))),
    ),
    repair_size=30,                   # an audio family's Res is the song length in seconds
    block_activity=True,              # the block sliders glow with each layer's push as the song plays
    repair_default_strength=0.5,      # long-run LoRAs are played at about half strength (the long-run hint)
    ema_default="0.98",               # Peter, 11 Oct 2026: the 800-epoch Official run with EMA 0.98 was far better
    ema_short_run=True,
    # Peter, 11 Oct 2026 (chiptune, Official preset): epoch 740 at strength 1.0 played one training song on every seed;
    # at 0.5 it kept the learned sound on new music, better than any earlier epoch at 1.0
    long_run_hint=(400, "a long run",
                   "After hundreds of epochs the LoRA has learned the fine detail of your songs, and at strength 1 it "
                   "can start copying them: the same song on every seed. Try the final LoRA at a lower strength, "
                   "around 0.5 - you keep the learned detail without the copying."),               # music sets are small: 10 clips x accumulation 4 is ~3 steps an epoch
    song_captions="chiptune, 8-bit, instrumental, square wave lead, arpeggios",
    sample_prompt="chiptune, 8-bit, instrumental, square wave lead, arpeggios",   # a song, not "A high quality photo"
    finetune=True,                    # the whole DiT trains, a window at a time on a 4-bit trunk (families/ft.py)
    ft_learning_rate=1e-5,
    ft_regularisation=False,          # no "broad class" photos for a song set
    # prompt sliders: a dial between two captions (calm <-> energetic), learned from the model's own songs - no
    # pairs (the same song recorded two ways is rare)
    kind_notes=(
        ("standard", "", "A normal LoRA: a sound, a genre or an artist's style, learned from your captioned songs."),
        ("slider", "Slider (a dial between two music styles)",
         "A dial: at strength +1 the LoRA moves a song toward one style, at -1 toward the other, and anything in "
         "between. Previews play -1, 0 and +1 one after another."),
        ("finetune", "", "The whole model learns your songs, not a LoRA on top of it. Each checkpoint is a full model "
                         "file (about 10 GB), loaded in place of the base."),
    ),
    slider_training=True,
    slider_sources=("prompts",),
    slider_prompt_notes=(
        ("FAMILY_SLIDER_BASE", "1. What the music is",
         "The start of the caption, e.g. \"instrumental, synth pop,\" - each end's words are added after it with a "
         "space, so end it with a comma. Keep it general and the dial works on any song of that kind."),
        ("FAMILY_SLIDER_POS", "2. The +1 end adds",
         "e.g. \"energetic, fast tempo, loud drums\"."),
        ("FAMILY_SLIDER_NEG", "3. The -1 end adds",
         "e.g. \"calm, slow tempo, soft\". Change one thing: words that also change the genre or the instruments "
         "turn the dial into a genre switch."),
        ("FAMILY_SLIDER_ANCHORS", "5. Leave these alone (optional)",
         "Captions of music the dial must not change, one a line. For a heavier-guitar dial meant for rock, "
         "\"instrumental, solo piano, classical\" keeps piano pieces as they are; for an energy dial, \"ambient, "
         "soft pads, no drums\" keeps ambient tracks calm. Each step checks one of them in turn, and the previews "
         "play the first one beside the dial, so you can hear it stay put. Each one makes a step about half as long "
         "again. Empty = none."),
        ("FAMILY_SLIDER_ANCHOR_WEIGHT", "6. How firmly",
         "How much keeping them as they are counts against the dial itself. 1 is equal weight; higher holds them "
         "tighter but can weaken the dial; lower lets a little of the change through."),
    ),
    ema_hint="Saves and previews the running average of the LoRA instead of its latest step. A song set gives few "
             "steps an epoch (10 songs at accumulation 4 is about 3), so on a short run 0.98 averages over most of "
             "it and lags behind; Short run sizes the window to a quarter of the run. Long runs suit 0.98.",

    lora=LoRAFormat(
        key_template="base_model.model.layers.{block}.{module}.{ab}.weight",
        down="lora_A", up="lora_B",
        block_modules=tuple(f"{a}.{p}" for a in ("self_attn", "cross_attn")
                            for p in ("q_proj", "k_proj", "v_proj", "o_proj")),
        alpha_key="{prefix}.alpha",
        file_prefix="base_model.model.",
        note="PEFT keys on the decoder (the official trainer's LoRA files), plus .alpha; ComfyUI maps "
             "base_model.model.<path> to diffusion_model.decoder.<path>. LoKR loads under the same names.",
        source="ComfyUI comfy/lora.py model_lora_keys_unet (ACEStep15)",
    ),

    driver="fizgig.acestep.driver:AceStepDriver",
    modelspec_arch="ACE-Step-1.5-XL",
    implementation=_REPO,
    precisions=("bf16",),
    optimizers=("adamw", "adamw8bit"),
    network_types=("lora",),          # LoRA only on the Training tab (Peter, 11 Oct 2026)
    adaptive_lr=False,
    loss_watch=False,
    helper_files=(("ACE-Step/Ace-Step1.5", ("Qwen3-Embedding-0.6B/*.json", "Qwen3-Embedding-0.6B/*.txt",
                                            "Qwen3-Embedding-0.6B/*.jinja", "acestep-v15-turbo/silence_latent.pt")),),

    sampling=(
        SamplingSettings("ACE-Step 1.5 XL Turbo", steps=8, cfg=1.0, sigmas=_TURBO_T + (0.0,),
                         note="The official Turbo schedule (shift 3, 8 steps), no CFG.",
                         source=f"{_REPO} acestep/models/turbo SHIFT_TIMESTEPS"),
    ),
    samples_cfg_free=True,
    preview_steps=8,
    preview_every=40,                 # a long run (800 epochs): a preview with every checkpoint
    # a long run is played at about half strength (Peter, 11 Oct 2026): each preview also renders at 0.5
    preview_strengths=(0.5,),
    preview_cfg=1.0,
    samples_text=(
        ("banner", "Each sample prompt is a song: \"caption | lyrics\", lyric lines separated by \" / \" (e.g. "
                   "\"[verse] / first line / second line\"). Leave out the lyrics for an instrumental. Songs train "
                   "for hundreds of epochs, so a preview every 25-50 epochs is plenty."),
    ),
    options=(
        FamilyOption("ACESTEP_TRAIN_BASE", "Training Base", tab="model", choices=(
            ("XL Turbo (8 steps) — standard", ""),
            ("XL SFT (30 steps + guidance)", "--dit=pref:acestep15xl_sft_dit base=sft")),
            hint="Turbo is what most people generate with. SFT is the non-distilled model: its previews take 30 "
                 "guided steps (slower) and it needs the SFT DiT in Preferences."),
        FamilyOption("ACESTEP_PREVIEW_SECONDS", "Sample length", tab="samples", choices=(
            ("10 seconds", "preview_seconds=10"), ("30 seconds", "preview_seconds=30"),
            ("60 seconds", "preview_seconds=60"), ("120 seconds", "preview_seconds=120")),
            default="60 seconds", hint="How long each preview song is."),
    ),

    presets=(
        # the official LoRA recipe (rank 64 / alpha 128, 1e-4, batch 1 x accumulation 4, the tutorial's 800 epochs for
        # 10-20 songs), measured in Fizgig 11 Oct 2026 (Peter, chiptune, 10 songs): the sound arrives after ~30 epochs,
        # epoch 740 played at strength 0.5 beat every earlier epoch at 1.0; a checkpoint + preview every 40 epochs
        ("✨ ACE-Step Official LoRA (rank 64, 800 epochs)", _preset(64, 128, 1e-4, 800)),
        # a prompt slider (11 Oct 2026, calm <-> energetic on synth pop - Peter: "brilliant"): rank 8, 2e-4, 20 epochs,
        # every practice song its own update (accumulation 1), a checkpoint and the dial's preview every 4 epochs
        ("✨ ACE-Step Slider (rank 8, 20 epochs)",
         {**_preset(8, 8, 2e-4, 20, ema="Off", save_every=4), "GRADIENT_ACCUMULATION": 1, "FAMILY_SLIDER": True,
          "FAMILY_SLIDER_GUIDANCE": "2"}),
        # the whole model: all five parts fit one window on a 32 GB card, so a rotation is an epoch there - 800 as the
        # LoRA recipe, a checkpoint (about 10 GB) and a preview every 50 (Peter, 11 Oct 2026)
        ("✨ ACE-Step Fine-tune (1e-5, 800 rotations)",
         {**_preset(64, 16, 1e-5, 800, ema="Off"), "FAMILY_FT": True, "FAMILY_FT_ROTATIONS": "800",
          "FAMILY_FT_SAVE_EVERY": "50"}),
    ),

    notes=(
        ("Licence: MIT (code and weights); generated music may be used commercially.", _REPO),
        ("A training song is an audio file with a caption .txt; its lyrics go in <name>.lyrics.txt (none = "
         "instrumental) and bpm / timesignature / keyscale / language in an optional <name>.json. Songs over 240 s "
         "are cut to the first 240 s (the official trainer's max_duration).", _REPO),
        ("Conditioning follows the official handler's text-to-music: the 'Fill the audio semantic mask' prompt with "
         "the metadata block, lyrics as '# Languages / # Lyric' token embeddings, the silence latent as context and "
         "as the 30 s timbre reference. The official trainer feeds lyrics raw and a zero timbre reference; Fizgig "
         "trains the way the model is prompted.", f"{_REPO} acestep/core/generation/handler"),
        ("XL's timbre encoder reads a learned CLS token placed before the reference frames (the XL code); ComfyUI "
         "reads the first frame without it, which moves the velocity ~2% with the silence reference (measured). "
         "Fizgig follows the official XL code.", "ACE-Step/acestep-v15-xl-turbo modeling_acestep_v15_xl_turbo.py"),
        ("Turbo trains on its own 8 timesteps (shift 3) with no CFG dropout, as the official trainer does. SFT trains "
         "with continuous logit-normal timesteps (mu -0.4, sigma 1) and 15% null-condition dropout (Side-Step's "
         "corrected mode) and previews with 30 steps, shift 1, APG guidance 7 (the XL SFT code's defaults).",
         "Side-Step docs: Model Management.md; ACE-Step/acestep-v15-xl-sft"),
    ),
)
