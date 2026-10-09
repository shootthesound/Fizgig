"""MiniMax H3 through the standard layer. It replaced the original H3 trainer (4 Oct 2026, src/fizgig/minimax/trainer.py
and scripts/minimax_train.py - in git history) once the two were shown to train the same: the same model files and
Preferences keys, the same arch id and cache layout (minimaxh3 - the dataset layer's clip / voice discovery keys on it,
and the 32B text caches are reused), kohya LoRA keys as the original writes them, and the original's Training-tab
settings as family options seeded from their old MINIMAX_* keys. Facts are the original trainer's, cited per value.
"""
from fizgig.families.description import (ClipSpec, FamilyDescription, FamilyOption, LoRAFormat, ModelFile,
                                        SamplingSettings, SpeedLoRA)

# The old H3 Training-tab controls as family options (lora_trainer_gui.py's MiniMax rows and _build_minimax_command),
# their old settings keys kept so presets, queued runs and Last Train carry across.
_H = "Comfy-Org/MiniMax-H3"     # the model files' Hugging Face repo
_OSTRIS = "pref:minimax_training_adapter[H3_TRAIN_BASE=ref2va|H3_DISTILL=1->minimax_ref_training_adapter]"
# Training Structure: the old dropdown, a view of the clean-end share (MINIMAX_LOWNOISE_PCT); Custom reveals the box
_STRUCTURE = (
    ("Likeness and Style — 60% clean-end", "lownoise_pct=60", "60",
     "Most of the run on nearly-clean images — the tuned default for stills. See the MiniMax section of the README."),
    ("Model default, movement — 8% clean-end", "lownoise_pct=8", "8",
     "The reference trainer's schedule, weighted to movement and composition. See the MiniMax section of the README."),
    ("Custom", "", "", "Type your own clean-end share. See the MiniMax section of the README."),
)


def _preset(rank, epochs, clip_still=True, slider=False, lr=1e-6, optimizer="automagic3"):
    """The old H3 built-ins (lora_trainer_gui.py MINIMAX_BUILT_IN_PRESETS), values unchanged, in the standard layer's
    keys and this family's option keys; slider=True the Slider preset (adamw8bit, as every slider trains)."""
    return {
        "NETWORK_DIM": rank, "NETWORK_ALPHA": rank, "NETWORK_TYPE": "LoRA (standard)", "LOKR_FACTOR": 8,
        "LEARNING_RATE": lr, "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "FAMILY_SLIDER": slider, "FAMILY_SLIDER_GUIDANCE": "2",
        "ADAPTIVE_LR": False, "ADAPTIVE_LR_MIN": "1e-5", "ADAPTIVE_LR_MAX": "4e-4", "OPTIMIZER_TYPE": optimizer,
        "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0, "DATASET_MEGAPIXELS": "0.25",
        "FAMILY_PRECISION": "Auto (recommended)", "BLOCKS_SWAP": "Auto (detect from GPU)",
        "FAMILY_EMA": "0.98 (recommended)",
        "H3_ADAPTER_RAMP": "Off", "H3_CAPTION_DROPOUT": "0.05 (default)", "H3_STRUCTURE": _STRUCTURE[0][0],
        "H3_LOWNOISE_PCT": "60", "H3_HIGHNOISE_LR_PCT": "100",
        "H3_BLOCKS": "all", "H3_TRAIN_REFINER": "", "H3_LIKENESS_MODE": "Default",
        "H3_ADAPTER": "Circlestone — best for photos", "H3_TREAD": "1", "H3_CLIP_STILL": "1" if clip_still else "",
        "H3_DISTILL": "",
    }


PRESETS = (
    ("✨ MiniMax H3 Fast (LoRA 8, 50 epochs)", _preset(8, 50)),
    ("✨ MiniMax H3 (rank 16, 60 epochs)", _preset(16, 60)),
    ("✨ MiniMax H3 Style (LoRA 8)", _preset(8, 50, clip_still=False)),
    # Slider (3 Oct): rank 8 (Peter) at 2e-4 for ~160 steps; a prompt smile slider at push 2
    # was clear by epoch 5 of 10 (16 practice pictures an epoch)
    ("✨ MiniMax H3 Slider (rank 8, 2e-4)", _preset(8, 16, clip_still=False, slider=True, lr=2e-4,
                                                     optimizer="adamw8bit")),
)

OPTIONS = (
    FamilyOption("H3_TRAIN_BASE", "Training Base", tab="model", choices=(
        ("First/last frame (fl2va) — standard", ""),
        ("Reference (ref2va)", "--dit=pref:minimax_ref_dit")),
        hint="Reference (ref2va) if the LoRA lives in the r2v workflow; needs DiT (reference) in Preferences.",
        setting="MINIMAX_TRAIN_BASE"),
    FamilyOption("H3_STRUCTURE", "Training Structure", choices=tuple((c[0], c[1]) for c in _STRUCTURE),
                 choice_values=tuple(c[2] for c in _STRUCTURE), choice_notes=tuple((c[0], c[3]) for c in _STRUCTURE),
                 setting="MINIMAX_LOWNOISE_PCT", hint_indent=7, pady=(8, 2), width=36),
    FamilyOption("H3_LOWNOISE_PCT", "Clean-end share", kind="entry", tokens="lownoise_pct={}", default="60", width=8, suffix="% of steps",
                 summary="low-noise {}%", setting="MINIMAX_LOWNOISE_PCT", requires="H3_STRUCTURE=Custom", pady=(2, 2)),
    FamilyOption("H3_HIGHNOISE_LR_PCT", "Medium to High Noise LR", kind="entry",
                 tokens="highnoise_lr_pct={}", default="100",
                 hint="Scales the LR of the noisy-half steps: pose, framing, face shape. Leave at 100 unless "
                      "experimenting.",
                 summary="high-noise LR {}%", setting="MINIMAX_HIGHNOISE_LR_PCT", width=8, suffix="%  — best left at 100 unless you are experimenting.", hint_indent=7, pady=(2, 8), hint_pady=(0, 8)),
    FamilyOption("H3_MIXED_STOP_CATEGORY", "Finish one category early", choices=(
        ("voice", "stop_category=audio"), ("photos & clips", "stop_category=visual")),
        setting="MIXED_STOP_CATEGORY", mixed_only=True, width=14, pady=(8, 2)),
    FamilyOption("H3_MIXED_STOP_EPOCH", " after epoch ", kind="entry", tokens="stop_epoch={}",
                 setting="MIXED_STOP_EPOCH", mixed_only=True, inline=True, width=5),
    FamilyOption("H3_MIXED_STOP_MODE", "", inline=True, width=26, choices=(
        ("anchor at 10% LR (recommended)", "stop_mode=anchor"), ("stop completely (faster)", "stop_mode=stop")),
        hint="Finish the smaller category early, before it overbakes. Blank = both train to the end. Anchor holds it "
             "at 10% LR and keeps its epoch report live. Stop skips its steps: faster, but unwatched.",
        setting="MIXED_STOP_MODE", mixed_only=True, hint_indent=7),
    FamilyOption("H3_LIKENESS_MODE", "Training mode", choices=(
        ("Default", "photo_blocks=20-49 clip_blocks=20-49 audio_blocks=20-49"),
        ("More Blocks", "--train_blocks=6-49"),
        ("Off · hand-pick the blocks in Other Options", "")),
        choice_hints=(
            ("Default", "High quality, versatile, best at preserving model priors. Photos, clips and voice all train "
                        "blocks 20-49, and the backward stops at the window so the steps are quicker too."),
            ("More Blocks", "Less preservation of model priors, high quality. May help when you are training a MOTION "
                            "concept specifically, since it reaches more of the model. It is not a likeness upgrade — "
                            "Default reaches higher likeness, sooner, with quicker steps — and Default may well be enough for motion too. Every step type trains 6-49, "
                            "at 44 blocks in the backward instead of 30. Blocks 0-5 stay out either way; they deform "
                            "anatomy and colour."),
            ("Off · hand-pick the blocks in Other Options", "The blocks are yours to pick, for experiments: Blocks to Train, in the Other Options section "
                                                   "further down this tab.")),
        summary="mode {}", setting="MINIMAX_LIKENESS_MODE", section="after", width=34, pady=(8, 2)),
    FamilyOption("H3_ADAPTER", "Training adapter", choices=(
        ("Circlestone — best for photos", "--training_adapter=pref:minimax_circlestone_adapter"),
        ("Ostris — best for videos", f"--training_adapter={_OSTRIS}"),
        ("Off", "")),
        hint="De-distills the base while your LoRA learns: frozen at 1.0 for every training step, off for previews "
             "and never in your saved file. Circlestone (one file for fl2va and ref2va) trains sharper LoRAs from "
             "photos; Ostris learns a video look faster. For mixed datasets, choose by whether the photos or the videos are "
             "the priority.",
        setting="MINIMAX_ADAPTER", section="after", compact=True, width=46),
    FamilyOption("H3_TREAD", "TREAD token routing — on clip steps, half the video tokens skip the middle blocks",
                 kind="check", tokens="tread=0.5@2-47", default="1",
                 hint="Faster clip steps: a random half of each clip's video tokens skips blocks 2-46 and rejoins "
                      "unchanged. Photos and clip stills always run in full; previews and your saved LoRA are "
                      "untouched. See the MiniMax section of the README.",
                 mode="lora", setting="MINIMAX_TREAD", show_if_media="clip", section="after"),
    FamilyOption("H3_CLIP_STILL", "Also train each clip's sharpest face frame as a photo", kind="check",
                 tokens="clip_still_as_photo=1 aux:clip_still=1", default="1",
                 hint="Each clip's sharpest face frame trains as a photo with the clip's caption. Picked at caching; "
                      "clips cached with this off use frame 0 until re-cached.",
                 setting="MINIMAX_CLIP_STILL", show_if_media="clip", section="after"),
    FamilyOption("H3_ADAPTER_RAMP", "Adapter-relative LR", kind="choice", choices=(
        ("Off", ""), ("0.003 (slow build)", "adapter_ramp=0.003"), ("0.005 (recommended)", "adapter_ramp=0.005"),
        ("0.01 (fast build)", "adapter_ramp=0.01")),
        hint="Makes the Learning Rate box a ceiling the run climbs toward. Set it where you want to end up.",
        setting="MINIMAX_ADAPTER_RAMP", section="other", width=24),
    FamilyOption("H3_CAPTION_DROPOUT", "Caption dropout", choices=(
        ("Off", "caption_dropout=0"), ("0.05 (default)", "caption_dropout=0.05"),
        ("0.10 (strong)", "caption_dropout=0.1")), choice_values=("0", "0.05", "0.1"), default="0.05 (default)",
        hint="Trains a few percent of steps with no caption, so the LoRA does not lean entirely on the trigger word.",
        setting="MINIMAX_CAPTION_DROPOUT", section="other", width=24),
    FamilyOption("H3_BLOCKS", "Blocks to Train", kind="entry", tokens="--train_blocks={}", default="all",
                 suggestions=("6-49 · recommended (skips 0-5)", "all · every block (50 of 50)",
                              "10-49 · skip the first 10", "14-37 · middle band", "25-49 · back half",
                              "0-24 · front half"),
                 hint="Train a subset of the 50 blocks. Type ranges and singles, comma-separated, like 3-12, 22, "
                      "31-33. Measured answers: 6-49 for the whole model (what More Blocks runs) and 20-49 for "
                      "likeness (Default). Blocks 0-5 are in neither: they deform anatomy and pull the dataset's "
                      "colour into the render.",
                 summary="blocks {}", setting="MINIMAX_BLOCKS", requires="H3_LIKENESS_MODE=Off", section="other", always_shown=True, counts_blocks=50, unmet_notes=(
                     ("Default", "Owned by the Training mode above: photos and clips 20-49, and voice the same. Set the mode to Off to hand-pick.", "photos and clips: 20-49"),
                     ("More Blocks", "Owned by the Training mode above: every step type trains 6-49. Set the mode to Off to hand-pick.", "every step type: 6-49")), pady=(8, 2)),
    FamilyOption("H3_DISTILL", "Learn identity from my dataset (reference distillation)", kind="check",
                 tokens="distill=1 aux:distill=1 --dit=pref:minimax_ref_dit",
                 hint="Experiment. Teaches the LoRA to reproduce identity the way H3 does from a reference photo. "
                      "Needs the ref2va model in Preferences.", summary="distill", setting="MINIMAX_DISTILL", section="other"),
    FamilyOption("H3_DISTILL_WEIGHT", "   teacher ", kind="entry", inline=True, always_shown=True, width=5,
                 suggestions=("0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"), tokens="distill_weight={}", default="0.8",
                 requires="H3_DISTILL", summary="teacher {}", setting="MINIMAX_DISTILL_WEIGHT", section="other"),
    FamilyOption("H3_DISTILL_REFS", "   references each ", kind="entry", tokens="aux:distill_refs={}", default="2",
                 inline=True, always_shown=True, width=4, suggestions=("1", "2", "3", "4"), requires="H3_DISTILL",
                 summary="{} refs", setting="MINIMAX_DISTILL_REFS", section="other"),
    FamilyOption("H3_DISTILL_PHASE1", "   identity-first ", choices=(
        ("Auto (from dataset size)", "distill_phase1=-1"), ("Off — blend throughout", "distill_phase1=0"),
        ("2 epochs", "distill_phase1=2"), ("4 epochs", "distill_phase1=4"), ("8 epochs", "distill_phase1=8"),
        ("16 epochs", "distill_phase1=16"), ("30 epochs", "distill_phase1=30")), requires="H3_DISTILL", setting="MINIMAX_DISTILL_PHASE1", section="other", inline=True, always_shown=True, width=22),
    FamilyOption("H3_TRAIN_REFINER", "Train the text token refiner", kind="check", tokens="train_token_refiner=1",
                 hint="Recommended off. Does not affect the ability to use a trigger word. The refiner sets how every "
                      "prompt is read; training it softens output and makes previews judder between epochs. LoRA and "
                      "fine-tune runs alike (under fine-tune it would train alongside every window, four times the "
                      "duty cycle of any block).",
                 summary="refiner on", setting="MINIMAX_TRAIN_REFINER", section="other"),
    FamilyOption("H3_SAMPLE_FRAMES", "Sample length", tab="samples", choices=(
        ("Still (1 frame)", "preview_frames=1"),
        ("22 frames with sound (~1s)", "preview_frames=22 preview_audio=1"),
        ("56 frames with sound (~2.3s)", "preview_frames=56 preview_audio=1"),
        ("124 frames with sound (~5s)", "preview_frames=124 preview_audio=1")), setting="SAMPLE_FRAMES"),
    FamilyOption("H3_FT_SCOPE", "Train on", choices=(("All media", ""), ("Photos only", "ft_scope=photo")),
                 hint="A dataset FILTER, not a mode. All media fine-tunes on everything in the folder - photos, clips, "
                      "voice. Photos only skips the clips and voice of a mixed folder (with Training mode on Default "
                      "the cycle then tightens to the identity blocks).",
                 setting="MINIMAX_FT_SCOPE", mode="finetune"),
    FamilyOption("H3_FT_BLOCKS", "Fine-tune blocks", kind="entry", tokens="ft_blocks={}",
                 hint="Optional: restrict the rotation cycle to a block range - the whole fine-tune touches only these "
                      "blocks. 20-49 is the measured likeness recipe (protects the fragile 0-19 trunk) and roughly "
                      "halves the system-RAM master copy. Empty = the full model.",
                 setting="MINIMAX_FT_BLOCKSPEC", mode="finetune"),
    FamilyOption("AUDIO_VAE", "", kind="fixed", tokens="audio_vae=pref:minimax_audio_vae aux:audio_vae=pref:minimax_audio_vae"),
)

MINIMAX = FamilyDescription(
    key="minimax",
    arch_id="minimaxh3",
    display_name="MiniMax H3",
    gui_label="MiniMax H3",
    lora_name_suffix="mmh3",
    experimental=True,
    hidden=False,

    # the Preferences rows (the old hand-built "Model Paths (MiniMax H3)" section, texts unchanged)
    model_files=(
        ModelFile("minimax_dit", "DiT", True, _H, "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                  21.0, role="dit",
                  hint="MiniMax H3 DiT — the training base. Use the PRUNED int8 file "
                       "(minimax_h3_fl2va_pruned_int8_convrot.safetensors, ~21 GB): it is the one ComfyUI runs, so your "
                       "LoRA trains against the weights it will be deployed on, and its curve-table AdaLN is a target a "
                       "LoRA can actually use. The ~66 GB bf16 file also works. The pruned file KEEPS its int8 weights "
                       "(~21 GB on the GPU, what the reference trainer does); the bf16 file is quantized to NF4 at load "
                       "(~11 GB, a little lossier).",
                  download_note="~21GB — Comfy-Org/MiniMax-H3 → diffusion_models/"
                                "minimax_h3_fl2va_pruned_int8_convrot.safetensors (fl2va is the trainable variant; the "
                                "66GB bf16 file works too)"),
        ModelFile("minimax_ref_dit", "DiT (reference)", False, _H,
                  "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors", 21.0, role="ref_dit",
                  hint="OPTIONAL — used when the Training tab's Training Base is set to Reference (ref2va), and for "
                       "reference distillation ('Learn identity from'). This is the ref2va model, a DIFFERENT "
                       "fine-tune from the fl2va one above and not just another quantization of it: it is what "
                       "ComfyUI's Reference-to-Video workflow loads, and the only H3 build that accepts reference "
                       "images. A LoRA trained on it is most faithful deployed on it. Leave blank if you only train on "
                       "the standard base.",
                  download_note="~21GB — Comfy-Org/MiniMax-H3 -> diffusion_models/"
                                "minimax_h3_ref2va_pruned_int8_convrot.safetensors (the pruned int8 build, same shape "
                                "as the fl2va one above; you may already have it if you use the r2v workflow)"),
        ModelFile("minimax_text_encoder", "Qwen3-VL-32B TE", True, _H,
                  "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", 15.7, role="text_encoder",
                  hint="Qwen3-VL-32B text encoder — nvfp4 (the compact ComfyUI file) or bf16 both work; the loader "
                       "detects which you gave it. The nvfp4 file keeps its packed weights (~15.7 GB on the GPU); bf16 "
                       "is NF4-quantized at load (~14 GB). Used only while caching caption embeddings, then offloaded "
                       "before training. (The int8_convrot TE variant is NOT supported — its rotated weights can't be "
                       "dequantized here.)",
                  download_label="Download nvfp4 (recommended)",
                  download_note="~15.7GB nvfp4-awq — the same TE ComfyUI uses, so you may already have it; identical "
                                "conditioning to bf16 (validated), just a slower one-off load",
                  alt_repo=_H, alt_path="text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors", alt_label="bf16",
                  alt_note="~51.5GB bf16 — the full-precision original; loads faster, 3.3x the disk"),
        ModelFile("minimax_vae", "Video VAE", True, _H, "vae/minimax_h3_video_vae_fp16.safetensors", 4.9, role="vae",
                  hint="The H3 video VAE — encodes each training image to a 24-channel latent (used only during "
                       "caching).",
                  download_note="~4.9GB — Comfy-Org/MiniMax-H3 → vae/minimax_h3_video_vae_fp16.safetensors"),
        ModelFile("minimax_audio_vae", "Audio VAE", False, _H, "vae/minimax_h3_audio_vae_fp32.safetensors", 0.61,
                  role="audio_vae", fetch_optional=False,
                  announce="Audio VAE (~605 MB) — train on the sound in video clips, and on voices",
                  hint="OPTIONAL — set this to train on the sound in your video clips. H3 generates audio and video "
                       "together, so a clip with sound can teach it a voice, and nothing else can. Used only during "
                       "caching, and only by clips: a folder of stills never loads it, and neither does a clip you "
                       "muted (a _mute on the filename trains that clip's video and ignores its sound). Leave blank "
                       "and clips train silent, exactly as they did before.",
                  download_note="~605MB — Comfy-Org/MiniMax-H3 → vae/minimax_h3_audio_vae_fp32.safetensors"),
        ModelFile("minimax_turbo_lora", "Turbo LoRA", False, "larryvrh/MiniMax-H3-Turbo-Lora",
                  "minimax_h3_turbo_v4_step600.safetensors", 0.78, role="speed_lora", fetch_optional=False,
                  announce="Turbo LoRA (~780 MB) — fast 6-step in-training previews",
                  hint="OPTIONAL — fast in-training previews. With this set, previews render in 6 steps with the "
                       "community Turbo LoRA applied at 75% on top of your training LoRA — the same pairing fast "
                       "ComfyUI inference uses — instead of the full 20-step pass. It touches PREVIEWS ONLY: the Turbo "
                       "is switched in for the sample render and out again before the next training step, and your "
                       "saved LoRA never contains it. Steps and strength are adjustable on the Samples tab.",
                  download_note="~780MB — larryvrh/MiniMax-H3-Turbo-Lora → minimax_h3_turbo_v4_step600.safetensors "
                                "(you may already have it in ComfyUI's loras folder)"),
        ModelFile("minimax_circlestone_adapter", "Training adapter (Circlestone)", False,
                  "circlestone-labs/MiniMax-H3-Image-Training-Adapter", "minimax_h3_image_training_adapter.safetensors",
                  0.62, role="training_adapter", fetch_optional=False,
                  announce="Training adapter (Circlestone, ~620 MB) — sharper, higher likeness",
                  hint="NEEDED when the Training adapter dropdown says Circlestone (the default) — one file for both "
                       "bases (fl2va and ref2va). A frozen LoRA that de-distills the base while yours learns: sharper "
                       "eyes, cleaner skin and better prompt-following than Ostris's on any dataset with stills. On for "
                       "every training step, off for previews, never in your saved LoRA. The updater fetches it; so "
                       "does the download button below.",
                  download_note="~620MB — circlestone-labs/MiniMax-H3-Image-Training-Adapter → "
                                "minimax_h3_image_training_adapter.safetensors"),
        ModelFile("minimax_training_adapter", "Training adapter (Ostris fl2va)", False,
                  "ostris/minimax_h3_training_adapter", "minimax_h3_training_adapter_v1.safetensors", 0.16,
                  fetch_optional=False,
                  hint="OPTIONAL — Ostris's adapter for the standard fl2va base, used when the Training tab's adapter "
                       "dropdown says Ostris: it learns a video look faster than Circlestone when the dataset is clips "
                       "only.",
                  download_note="~155MB — ostris/minimax_h3_training_adapter → minimax_h3_training_adapter_v1.safetensors"),
        ModelFile("minimax_ref_training_adapter", "Training adapter (Ostris ref2va)", False,
                  "ostris/minimax_h3_training_adapter", "minimax_h3_ref2va_training_adapter_v1.safetensors", 0.16,
                  fetch_optional=False,
                  hint="OPTIONAL — Ostris's adapter for the Reference (ref2va) base: picked automatically when the "
                       "dropdown says Ostris and the Training Base is ref2va or the run is a distillation run.",
                  download_note="~155MB — ostris/minimax_h3_training_adapter → "
                                "minimax_h3_ref2va_training_adapter_v1.safetensors"),
    ),
    prefs_title="Model Paths (MiniMax H3)",
    prefs_intro="Image-only LoRA training for MiniMax's ~33B H3 omni DiT. Train on the pruned int8 DiT — the same file "
                "ComfyUI runs — quantized to NF4 at load, so the resident base is ~11 GB. The Qwen3-VL-32B text "
                "encoder and the video VAE are only needed for the one-time caching pass; the compact nvfp4 TE is "
                "recommended. Trains on stills, or on short video clips — and with the audio VAE set, on their sound "
                "too.",
    fetch_note="Fetches the DiT, text encoder, both VAEs, the Turbo LoRA and the three training adapters above, plus "
               "the Krea 2 Qwen3-VL captioning text encoder (~47 GB all in), and fills in these paths for you — plus "
               "the small helper models (Florence-2 captioner, face model for the Look Filter and likeness scoring, "
               "EN→ZH translator, Gizmo's Whisper transcriber — ~1.9 GB) so nothing stalls to download later and "
               "everything works offline. No HuggingFace account needed — none of these are gated. The reference DiT "
               "is left out unless you tick it above: another 21 GB, and it is only used by identity mode.",
    fetch_optional_label="Include the reference DiT (+21 GB)",
    announce_intro="Fizgig can now train on video, sound and voices, and render fast Turbo previews.",
    text_encoder_label="Qwen3-VL-32B",
    vae_label="MiniMax H3 Video VAE",

    options=OPTIONS,
    presets=PRESETS,
    settings_aliases={"FAMILY_EMA": "MINIMAX_EMA", "FAMILY_PRECISION": "MINIMAX_BASE_QUANT", "FAMILY_FT": "MINIMAX_FINETUNE",
                      "FAMILY_FT_ROTATE_EVERY": "MINIMAX_FT_EVERY", "FAMILY_FT_FUSED": "MINIMAX_FT_FUSED",
                      "FAMILY_FT_REG_DIR": "MINIMAX_REG_DIR", "FAMILY_FT_REG_MULT": "MINIMAX_REG_MULT"},
    workbench=("repair", "explorer", "profiler", "extract", "royale"),
    workbench_engine="fizgig.minimax.workbench:H3WorkbenchEngine",
    finetune=True,
    ft_learning_rate=3e-5,            # the tested H3 fine-tune rate (1e-4 destroys; 1e-5 too slow to judge from)                    # the old rotation FT (component windows on an NF4 trunk, int8 ConvRot saves)
    media=("photo", "clip", "voice"),
    model_note="Previews default to 768×768 56-frame clips with sound; Sample length has stills and other lengths. "
               "📖 Full write-ups in the README.",
    samples_text=(
        ("banner", "Preview prompts rendered periodically during training, as short clips on the model being "
                   "trained. Samples land in <output_dir>/sample/ and the Gallery button below opens the viewer."),
        ("advanced", "H3 renders CFG-free on a fixed schedule, so the knobs it does not use are greyed out."),
        ("flow", "Fixed at 12 for H3 — the schedule every shipped workflow uses"),
        ("neg", "Unused — H3 samples render without CFG"),
        ("cfg", "H3 renders without CFG; the shipped workflow does the same"),
        ("steps", "20 steps matches the shipped H3 workflow")),
    samples_cfg_free=True,
    preset_notes=(("✨ MiniMax H3 (rank 16, 60 epochs)", "(more suitable for larger datasets with longer trains)"),),
    multi_concept=True,
    # the old H3 recipe on the box's click: a stronger caption dropout (each subject must answer to its own trigger)
    multi_concept_defaults=(("H3_CAPTION_DROPOUT", "0.10 (strong)"),),
    multi_concept_hint="See the MiniMax section of the README.",
    int8_attention=True,              # workbench renders: comfy-kitchen's INT8 attention
    activation_cache=True,            # Turbo Preview: step-1 replay, identical to a full render
    adaptive_lr=False,
    slider_training=True,             # prompt pairs, photo pairs and clip pairs (driver: _slider_loss / predict)
    loss_watch=False,
    network_hint="LoRA recommended for MiniMax",
    ema_hint="A smoothed average of the weights, leading to better and more reliable previews.",
    samples_turbo_pace=True,
    ema_section="other",
    precision_label="Base Precision",
    precision_after_states=True,
    preview_park_optimizer=True,      # the old previews' optimizer-state park (~2.5 GB back for the render)
    # H3's clock (24 fps), its VAE's 17n+5 frame grid (5 ... 124) and /32 edges, its 32 kHz stereo audio VAE
    # Dial = 4 steps at Turbo 1.0 (the fast loop), Confirm = 6 at 0.75 (the render that matches training previews)
    clip_regimes=(("dial", 4, 1.0), ("confirm", 6, 0.75)),
    clip_spec=ClipSpec(fps=24, frame_step=17, frame_offset=5, max_frames=124, edge_multiple=32, audio_rate=32000,
                       audio_channels=2, mute_suffix="_mute", note="minimax/model.py FPS, pixel_frames_for_latent"),

    latent_channels=24,
    spatial_factor=16,                # dataset/image_dataset.py LATENT_SPATIAL_FACTOR[minimaxh3]
    bucket_step=32,                   # BUCKET_RESO_STEPS[minimaxh3]: 16x VAE x 2x2 patch
    image_channels=3,
    native_megapixels=0.6,            # 768 short edge (the old Samples default)

    n_blocks=50,                      # MiniMaxH3Config.num_layers
    block_prefix="blocks",
    block_note="Block ids follow the old Repair Studio (h3blk_0-49, h3_rf_0-1), so its saved presets apply.",

    lora=LoRAFormat(
        key_template="lora_unet_blocks_{block}_{module}.{ab}.weight",
        down="lora_down", up="lora_up",
        block_modules=("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"),
        alpha_key="{prefix}.alpha",
        kohya=True,
        file_prefix="lora_unet_",
        note="kohya keys as the old H3 trainer writes them (create_network(None, 'lora_unet', ...)); the AdaLN "
             "projection is left out by default (--no_train_adaln).",
        source="the original H3 trainer (minimax/trainer.py train_minimax, removed 4 Oct 2026 - in git history)",
    ),

    driver="fizgig.minimax.driver:MiniMaxDriver",
    modelspec_arch="MiniMax-H3",
    implementation="https://github.com/MiniMax-AI/MiniMax-H3",
    ema_default="0.98",               # the old H3 default (v5.4.1)
    ema_short_run=True,
    resumes_untagged_states=True,     # an old H3 pause: same files, same parameter order (blocks, qkv/out/fc1/fc2)
    # the training adapter is the MINIMAX_ADAPTER option (Circlestone / Ostris per base / Off), not the generic tick
    # the checkpoint's own int8 ConvRot codes, or a 4-bit base (NF4, HQQ); Auto is the driver's plan_run (the old
    # planner), which streams blocks H2D rather than giving up the int8 base
    precisions=("int8", "nf4", "hqq"),
    precision_labels={"auto": "Auto (recommended)", "int8": "int8 · most accurate, needs ~30 GB free",
                      "nf4": "4-bit · fits smaller cards", "hqq": "4-bit HQQ · lower error than 4-bit, slower"},
    precision_hint="Auto reads free VRAM at launch and picks precision and block swap together. int8 is the most "
                   "accurate, 4-bit fits smaller cards, 4-bit HQQ sits between (less error, more VRAM, slower with no "
                   "swap). Auto never picks HQQ.",
    optimizers=("automagic3", "adamw8bit", "adamw"),
    optimizer_weight_decay=1e-4,      # the old trainer's (ai-toolkit's job template); bnb's default is 1e-2
    optimizer_eps_floor_8bit=True,
    trainable_dtype="bf16",           # the old trainer: network.to(device, dtype=bfloat16)
    network_types=("lora", "lokr"),

    sampling=(
        SamplingSettings("H3 reference", steps=20, cfg=1.0, sampler="euler", scheduler="simple",
                         note="CFG-free on the fixed shift-12 schedule, as the shipped ComfyUI workflows.",
                         source="lora_trainer_gui.py ARCHITECTURES['MiniMax H3'] sample defaults"),
    ),
    speed_loras=(
        SpeedLoRA(
            name="H3 Turbo LoRA (6-step)",
            repo="larryvrh/MiniMax-H3-Turbo-Lora",
            file="minimax_h3_turbo_v4_step600.safetensors",
            pairs_with="MiniMax H3 fl2va",
            strength=0.75,
            settings=SamplingSettings("Turbo 6-step", steps=6, cfg=1.0, sampler="euler", scheduler="simple",
                                      note="CFG-free; the old previews' 6 steps at 0.75.",
                                      source="the original H3 trainer (minimax/trainer.py load_preview_turbo, in git history)"),
            load_unmerged=True,
            pref_key="minimax_turbo_lora",
            source="https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora",
        ),
    ),
    preview_speed_lora="H3 Turbo LoRA (6-step)",
    preview_steps=20,
    preview_cfg=1.0,
    preview_width=768,
    preview_height=768,
)
