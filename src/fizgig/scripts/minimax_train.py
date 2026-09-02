"""MiniMax H3 image-only LoRA training CLI (wraps fizgig.minimax.trainer.train_minimax).

The GUI drives the full run as three steps: minimax_cache_latents -> minimax_cache_text ->
minimax_train. Trains a LoRA over an NF4-quantized frozen base (QLoRA). No samples, no preview.
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# CUDA allocator policy, set before torch is imported below (the backend is fixed at CUDA init).
# The 33B base + per-step tensor churn fragments the default allocator; expandable segments hold
# the NF4 base within a 5090's budget. GUI sets this too; this covers headless runs.
if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF") and os.environ.get("FIZGIG_NO_EXPANDABLE") != "1":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

from fizgig.minimax.trainer import train_minimax
from fizgig.training.optimizers import available_optimizers

logging.basicConfig(level=logging.INFO)


def _shift_arg(v):
    """--shift takes 'sigmoid' / 'resolution', 'lognorm:<float>', or a plain float.

    'lognorm:<s>' is the same shift map as a plain float but drawn from a logit-normal base
    instead of a uniform one — the SHAPE axis: mid-concentrated rather than flat with fat tails.
    """
    if v in ("sigmoid", "resolution"):
        return v
    if isinstance(v, str) and v.startswith("lognorm:"):
        float(v.split(":", 1)[1])          # validate now, not 20 minutes into a run
        return v
    return float(v)


def setup_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MiniMax H3 image-only LoRA training (NF4 base, no samples)")
    p.add_argument("--dit", required=True, help="H3 bf16 DiT (minimax_h3_fl2va_bf16.safetensors)")
    p.add_argument("--dataset_config", required=True, help="Dataset .toml")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_name", required=True)
    p.add_argument("--network_dim", type=int, default=16)
    p.add_argument("--network_alpha", type=float, default=16)
    p.add_argument("--network_type", default="lora", choices=["lora", "lokr"],
                   help="Trainable parametrization: standard LoRA, or LoKR (Kronecker, "
                        "full-matrix w2 — dim/alpha unused, --lokr_factor is the dial)")
    p.add_argument("--lokr_factor", type=int, default=8,
                   help="LoKR only: Kronecker split factor (w1 is ~factor x factor)")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_train_epochs", type=int, default=10)
    p.add_argument("--save_every_n_epochs", type=int, default=0)
    p.add_argument("--save_state", action="store_true",
                   help="Write a resumable <name>-NNNNNN-state/ dir at every checkpoint.")
    p.add_argument("--save_state_on_train_end", action="store_true",
                   help="Write a resumable state dir when the run finishes, so a completed LoRA "
                        "can be trained further by raising --max_train_epochs.")
    p.add_argument("--keep_last_n_states", type=int, default=2,
                   help="Keep only the N newest state dirs.")
    p.add_argument("--resume", default=None,
                   help="Resume from a saved <name>-NNNNNN-state dir (optimizer + RNG + "
                        "adaptive-LR state restored).")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--optimizer_type", default="adamw8bit", choices=available_optimizers())
    p.add_argument("--optimizer_args", default="", help='Free-form kwargs, e.g. "weight_decay=0.01"')
    p.add_argument("--caption_dropout", type=float, default=0.05,
                   help="Fraction of steps trained on the empty prompt (reference default 0.05; "
                        "needs the uncond embed cached by minimax_cache_text). 0 disables.")
    p.add_argument("--audio_weight", type=float, default=1.0,
                   help="Weight on the audio term for video clips that carry sound. Audio is "
                        "only ~4%% of the packed sequence, so parity may be too quiet to teach "
                        "anything — raise it if the per-epoch [audio] line shows sound winning "
                        "a negligible share of the loss. Ignored by stills and muted clips.")
    p.add_argument("--visual_stop_epoch", type=int, default=0,
                   help="Mixed datasets: retire photos & clips after this epoch (0 = never). "
                        "See --visual_stop_mode for what retirement means.")
    p.add_argument("--visual_stop_mode", choices=["anchor", "stop"], default="anchor",
                   help="anchor = the retired category keeps training at 10%% LR (holds its "
                        "quality against drift on the shared adapters, and its epoch ledger "
                        "stays live as the drift alarm); stop = skipped outright (faster "
                        "epochs, blind).")
    p.add_argument("--audio_stop_epoch", type=int, default=0,
                   help="Mixed datasets: retire voice recordings after this epoch (0 = never).")
    p.add_argument("--audio_stop_mode", choices=["anchor", "stop"], default="anchor",
                   help="As --visual_stop_mode, for the voice recordings.")
    p.add_argument("--include_patterns", nargs="*", default=None,
                   help="Regex module filters (Model Area to Train). Default: all transformer blocks.")
    p.add_argument("--distill", action="store_true",
                   help="EXPERIMENT: reference distillation. Instead of only reconstructing each "
                        "photo, the LoRA is taught to behave — from text alone — as the model "
                        "does when SHOWN a reference. The references are OTHER images from this "
                        "same dataset, paired by minimax_cache_text --reference_count. The whole "
                        "run trains on ref2va (it is the only build that accepts references), "
                        "but the resulting LoRA deploys fine on the ordinary fl2va model.")
    p.add_argument("--distill_weight", type=float, default=0.8, metavar="W",
                   help="Teacher share of the loss (default 0.8); the remaining 1-W is the "
                        "ordinary flow loss against the real photo. 1.0 is pure distillation, "
                        "which caps the LoRA at exactly the teacher's habits; a little photo "
                        "keeps real photographic detail reachable. Ignored when "
                        "--distill_phase1_epochs is active.")
    p.add_argument("--distill_phase1_epochs", type=int, default=-1, metavar="N",
                   help="Identity-first: train the first N epochs against the TEACHER ONLY, then "
                        "drop the teacher and train on the photographs alone. -1 (default) picks "
                        "N from the dataset size so phase 1 is ~650 gradient steps, which is "
                        "where the teacher objective converges. 0 = off (blended loss "
                        "throughout, using --distill_weight).")
    p.add_argument("--slow_blocks", default=None, metavar="SPEC",
                   help="EXPERIMENT: train these blocks at a reduced learning rate (same syntax "
                        "as --train_blocks). A perturbation in a late block reaches the output "
                        "almost undamped while an early one is absorbed by everything after it, "
                        "so one LR is too high for the late blocks or too low for the early ones. "
                        "Pair with --slow_block_lr_scale.")
    p.add_argument("--slow_block_lr_scale", type=float, default=1.0, metavar="X",
                   help="LR multiplier for --slow_blocks (e.g. 0.2 = one fifth). 1.0 disables.")
    p.add_argument("--block_limit", type=float, default=0.0,
                   help="Per-step movement clip: cap any block's movement in a SINGLE step at "
                        "N x the median block's step (1.25 recommended). Prevents the overshoot "
                        "rather than undoing accumulated learning. 0 = off.")
    p.add_argument("--adapter_ramp", type=float, default=0.0, metavar="R",
                   help="Adapter-relative LR: hold each step at fraction R of the adapter's "
                        "current size, so the LR starts low and climbs toward --learning_rate "
                        "as the adapter grows (0.005 = ~0.5%% growth per step). Dimensionless, "
                        "so it needs no dataset or network-type calibration. 0 = off.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1, metavar="N",
                   help="Sum the gradient over N batches before each optimizer step (effective "
                        "batch N). Same wall-clock per epoch; each step is aimed by N images "
                        "instead of one, so a large stride carries far less sampling noise.")
    p.add_argument("--lr_warmup_epochs", type=float, default=0.0, metavar="N",
                   help="Ramp the LR linearly from 0 to the configured rate over the first N "
                        "epochs (fractions allowed). Static LR only — ignored under adaptive. "
                        "0 = off.")
    p.add_argument("--ema_decay", type=float, default=0.0, metavar="D",
                   help="Keep an exponential moving average of the adapter and save/preview THAT "
                        "instead of the raw weights (0.99 recommended). Training still runs on "
                        "the raw weights. 0 = off.")
    p.add_argument("--no_train_adaln", dest="train_adaln", action="store_false",
                   help="EXPERIMENT: drop the per-block AdaLN adapters. AdaLN is a function of "
                        "the TIMESTEP only, so it cannot encode identity — yet on the pruned "
                        "checkpoint it carries ~45%% of all weight movement. Turning it off frees "
                        "that capacity for the attention and MLP paths, which do see the image. "
                        "No effect on the bf16 checkpoint (it never targets AdaLN).")
    p.add_argument("--train_blocks", default=None, metavar="SPEC",
                   help="EXPERIMENT: train only these DiT blocks (of 50) instead of all of them. "
                        "Ranges and singles, comma-separated: '14-37' or '3-12, 14-15, 22, 31-33'. "
                        "The text refiner is always included. H3's blocks are identical and nobody "
                        "has mapped what each one does, so any selection is a hypothesis — compare "
                        "against a full-model run on the same dataset.")
    p.add_argument("--photo_blocks", default=None, metavar="SPEC",
                   help="Optimised Likeness Learning: photo training steps update only these "
                        "DiT blocks (the refiner always trains); video/audio clip steps update "
                        "the full model. '20-49' is the measured likeness recipe — photo "
                        "gradients into the front trunk erode rendering and anatomy while "
                        "identity lives in the back blocks. Composes with --train_blocks.")
    p.add_argument("--audio_blocks", default=None, metavar="SPEC",
                   help="Voice routing: audio-only training steps update only these DiT "
                        "blocks (the refiner always trains). '34-49' is the measured voice "
                        "zone (core 38-48 + shoulder) — audio gradients outside it "
                        "measurably corrupt the visual blocks (A/B, 24 Aug). Applies "
                        "under the rotation fine-tune, and in LoRA mode alongside "
                        "--photo_blocks.")
    p.add_argument("--clip_blocks", default=None, metavar="SPEC",
                   help="Fine-tune only: confine VIDEO CLIP training steps to these DiT "
                        "blocks (the refiner always trains). The GUI's 'Restrict video to "
                        "likeness blocks' passes the likeness set here — a confined "
                        "overnight video run trained perfectly well (field, 29 Aug). "
                        "Unset: clips train the full model, the original behaviour.")
    p.add_argument("--base_quant", default="auto", choices=["auto", "int8", "nf4", "hqq"],
                   help="Frozen-base precision. 'int8' keeps the checkpoint's own ConvRot "
                        "weights (~0.17%% base error, ~21 GB) — what the reference trainer "
                        "does. 'nf4' decodes then 4-bit quantizes (~9.5%% error, ~11 GB). "
                        "'hqq' decodes then HQQ 4-bit quantizes (~6.3%% error, ~15 GB, ~half "
                        "the step speed). 'auto' picks int8 for a pre-quantized "
                        "file, nf4 otherwise — never hqq.")
    p.add_argument("--no_quantize", action="store_true",
                   help="Train on the bf16 base (no NF4) — needs ~66 GB VRAM.")
    p.add_argument("--blocks_to_swap", default="auto",
                   help="'auto' (plan from free VRAM + bucket MP) or an integer — park the last "
                        "N blocks on CPU between uses. 0 disables. Auto picks 0 on 32 GB cards.")
    p.add_argument("--gradient_checkpointing", default="auto", choices=["auto", "on", "off"],
                   help="Recompute blocks in backward to cut activation VRAM. Auto: off when "
                        "everything fits (faster), on otherwise; forced on when swap > 0.")
    p.add_argument("--shift", type=_shift_arg, default=None,
                   help="Training timestep density. Unset (recommended) = H3's own recipe: the "
                        "shift-12 uniform map (what the reference trainer uses). 'sigmoid' = "
                        "unshifted logit-normal and 'resolution' = resolution-shifted "
                        "logit-normal, both A/B modes that overdrive adapters at 1e-4. "
                        "A float = the uniform-u shift map at that value.")
    p.add_argument("--highnoise_lr_scale", type=float, default=1.0,
                   help="Learning-rate multiplier for steps drawn ABOVE sigma 0.5 — the noisy "
                        "half, where pose and composition are decided. 1.0 (default) leaves them "
                        "alone. Lower it as --shift moves the run toward the noisy end, where "
                        "those steps become the majority and would otherwise swamp the few "
                        "clean-end steps that carry identity.")
    # Adaptive LR — bi-directional plateau tracker (starts at the geometric midpoint of min/max;
    # the Learning Rate box is ignored while it's on).
    p.add_argument("--adaptive_lr", action="store_true")
    p.add_argument("--adaptive_lr_min", type=float, default=1e-5)
    p.add_argument("--adaptive_lr_max", type=float, default=4e-4)
    # In-training previews (Samples tab). Prompts file: one prompt per line, # comments ignored.
    p.add_argument("--sample_prompts", default=None, help="Prompt file, one per line")
    p.add_argument("--sample_every_n_epochs", type=int, default=0)
    p.add_argument("--sample_at_first", action="store_true", help="Render an epoch-0 preview")
    p.add_argument("--sample_width", type=int, default=768,
                   help="H3's native canvas is a 768 short edge (768*1344 pixel cap)")
    p.add_argument("--sample_height", type=int, default=768)
    p.add_argument("--sample_steps", type=int, default=20,
                   help="Denoise steps per preview. 20 matches ComfyUI's shipped MiniMax "
                        "template (BasicScheduler simple/20), and previews use the same "
                        "res_multistep sampler and sigma grid, so the count means the same "
                        "thing in both places.")
    p.add_argument("--sample_cfg_scale", type=float, default=1.0,
                   help=">1 enables CFG (a 2nd forward per step); the shipped H3 workflows use none")
    p.add_argument("--sample_frames", type=int, default=1,
                   help="Pixel frames per sample on the 17n+5 grid (1=still; 124=trained minimum, ~5s). Off-grid snaps down.")
    p.add_argument("--sample_negative", default=None, help="Only used when --sample_cfg_scale > 1")
    p.add_argument("--sample_seed", type=int, default=42, help="0 = random each preview")
    p.add_argument("--turbo_lora_path", default=None,
                   help="H3 Turbo LoRA (minimax_h3_turbo_v4_step600.safetensors) — applied to "
                        "PREVIEWS only, at --turbo_lora_strength, and removed before the next "
                        "training step. Pair with --sample_steps 6.")
    p.add_argument("--turbo_lora_strength", type=float, default=0.75,
                   help="Preview strength for --turbo_lora_path (0.75 recommended)")
    p.add_argument("--context_lora_path", default=None,
                   help="Context LoRA: an existing H3 LoRA loaded FROZEN and ACTIVE under the "
                        "trainable one, in training and previews. The output is trained to be "
                        "paired with it at inference. Not available with --finetune_rotation.")
    p.add_argument("--context_lora_strength", type=float, default=1.0,
                   help="Strength the context LoRA rides at (0.0-2.0)")
    p.add_argument("--training_adapter_path", default=None,
                   help="Training adapter (Ostris, ostris/minimax_h3_training_adapter): a frozen "
                        "LoRA at 1.0 that de-distills the base while yours learns — on for every "
                        "training step, off for previews. Use the fl2va or ref2va file to match "
                        "--dit. Not available with --finetune_rotation.")
    p.add_argument("--sample_audio", action="store_true",
                   help="Clip previews carry their generated SOUND: the jointly-denoised "
                        "audio rows are decoded to a .wav beside each sample. Needs "
                        "--audio_vae. Stills are unaffected.")
    p.add_argument("--audio_vae", default=None,
                   help="H3 audio VAE (minimax_h3_audio_vae_fp32.safetensors) — its decoder "
                        "half, for --sample_audio")
    p.add_argument("--text_encoder", default=None,
                   help="Qwen3-VL-32B TE — needed only to pre-encode preview prompts")
    p.add_argument("--vae", default=None, help="H3 video VAE (reserved for the full decoder)")
    # Output metadata
    p.add_argument("--metadata_title", default=None)
    p.add_argument("--metadata_author", default=None)
    p.add_argument("--metadata_description", default=None)
    p.add_argument("--metadata_license", default=None)
    p.add_argument("--metadata_tags", default=None)
    p.add_argument("--metadata_trigger_phrase", default=None)
    # ---- rotation full fine-tune (trains the BASE, not a LoRA) ----
    p.add_argument("--finetune_rotation", type=int, default=0,
                   help="Master switch: >0 trains the base model itself in component "
                        "windows. The output is a full ~21 GB int8 checkpoint per save.")
    p.add_argument("--finetune_rotate_every", type=int, default=1,
                   help="Advance the window every N epochs")
    p.add_argument("--finetune_rotation_mode", default="component",
                   choices=["component"],
                   help="component (the only mode) trains one "
                        "matmul (qkv/out/fc1/fc2) across EVERY block per window, on an "
                        "NF4-resident base — full model depth each epoch; the saved "
                        "checkpoint is still exact int8.")
    p.add_argument("--finetune_start_window", type=int, default=0,
                   help="Continue a fine-tune mid-cycle (printed at every save)")
    p.add_argument("--finetune_fused_backward", action="store_true", default=True,
                   help="Free each gradient as it lands (per-tensor optimizers; "
                        "disables grad clipping and accumulation)")
    p.add_argument("--no_finetune_fused_backward", dest="finetune_fused_backward",
                   action="store_false")
    p.add_argument("--finetune_scope", default="all", choices=["all", "photo"],
                   help="photo = skip clip/voice batches; all = the full mixed regime")
    p.add_argument("--finetune_blocks", default=None,
                   help="Restrict the rotation cycle to a block spec (e.g. '20-49', "
                        "the likeness recipe) — shrinks the CPU master too")
    p.add_argument("--finetune_master", default="auto", choices=["auto", "ram", "disk"],
                   help="Where the bf16 master lives. ram = whole master resident "
                        "(field-proven, ~0.77 GB/block). disk = lazy reads from the int8 "
                        "file + trained tensors spilled to scratch, ~one tensor in RAM — "
                        "what fits full-model FT on 64 GB boxes. auto picks disk when the "
                        "master would eat over 40%% of available RAM.")
    p.add_argument("--finetune_scratch_dir", default=None,
                   help="disk master's spill directory (wants a fast local drive). "
                        "Default: beside the dataset caches.")
    p.add_argument("--reg_lr_multiplier", type=float, default=0.2,
                   help="Fine-tune only: LR multiplier for images in a dataset block marked "
                        "`is_reg = true`. They anchor the model's prior rather than teaching "
                        "a subject, so they train as a nudge (0.1-0.3). They follow the photo "
                        "routing and stop with the visual category. Ignored when no reg block "
                        "is present; ignored (with a warning) on LoRA runs.")
    return p


def _read_prompts(path):
    """One prompt per line; blanks and # comments skipped. None when unset/missing."""
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        out = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    return out or None


def main():
    from fizgig.utils.device import apply_sim_vram_cap
    apply_sim_vram_cap()          # FIZGIG_SIM_VRAM_GB: behave like a smaller card
    args = setup_parser().parse_args()
    train_minimax(
        dataset_config=args.dataset_config,
        output_dir=args.output_dir,
        output_name=args.output_name,
        dit_path=args.dit,
        network_dim=args.network_dim,
        network_alpha=args.network_alpha,
        network_type=args.network_type,
        lokr_factor=args.lokr_factor,
        learning_rate=args.learning_rate,
        max_train_epochs=args.max_train_epochs,
        save_every_n_epochs=args.save_every_n_epochs,
        save_state=args.save_state,
        save_state_on_train_end=args.save_state_on_train_end,
        keep_last_n_states=max(1, args.keep_last_n_states),
        resume_state_dir=args.resume,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        optimizer_type=args.optimizer_type,
        optimizer_args=args.optimizer_args,
        caption_dropout=args.caption_dropout,
        audio_weight=args.audio_weight,
        visual_stop_epoch=args.visual_stop_epoch,
        visual_stop_mode=args.visual_stop_mode,
        audio_stop_epoch=args.audio_stop_epoch,
        audio_stop_mode=args.audio_stop_mode,
        base_quant=args.base_quant,
        include_patterns=args.include_patterns,
        train_blocks=args.train_blocks,
        photo_blocks=args.photo_blocks,
        clip_blocks=args.clip_blocks,
        audio_blocks=args.audio_blocks,
        distill=args.distill,
        distill_weight=args.distill_weight,
        distill_phase1_epochs=args.distill_phase1_epochs,
        train_adaln=args.train_adaln,
        slow_blocks=args.slow_blocks,
        slow_block_lr_scale=args.slow_block_lr_scale,
        block_limit=args.block_limit,
        adapter_ramp=args.adapter_ramp,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_warmup_epochs=args.lr_warmup_epochs,
        ema_decay=args.ema_decay,
        quantize=not args.no_quantize,
        shift=args.shift,
        highnoise_lr_scale=args.highnoise_lr_scale,
        blocks_to_swap=args.blocks_to_swap,
        gradient_checkpointing=args.gradient_checkpointing,
        adaptive_lr=args.adaptive_lr,
        adaptive_lr_min=args.adaptive_lr_min,
        adaptive_lr_max=args.adaptive_lr_max,
        metadata_title=args.metadata_title,
        metadata_author=args.metadata_author,
        metadata_description=args.metadata_description,
        metadata_license=args.metadata_license,
        metadata_tags=args.metadata_tags,
        metadata_trigger_phrase=args.metadata_trigger_phrase,
        sample_prompts=_read_prompts(args.sample_prompts),
        te_path=args.text_encoder,
        vae_path=args.vae,
        sample_every_n_epochs=args.sample_every_n_epochs,
        sample_at_first=args.sample_at_first,
        sample_width=args.sample_width,
        sample_height=args.sample_height,
        sample_steps=args.sample_steps,
        sample_cfg_scale=args.sample_cfg_scale,
        sample_frames=args.sample_frames,
        sample_negative=args.sample_negative,
        sample_seed=args.sample_seed,
        turbo_lora_path=args.turbo_lora_path,
        turbo_lora_strength=args.turbo_lora_strength,
        context_lora_path=args.context_lora_path,
        context_lora_strength=args.context_lora_strength,
        training_adapter_path=args.training_adapter_path,
        sample_audio=args.sample_audio,
        audio_vae_path=args.audio_vae,
        finetune_rotation=args.finetune_rotation,
        finetune_rotate_every=args.finetune_rotate_every,
        finetune_rotation_mode=args.finetune_rotation_mode,
        finetune_start_window=args.finetune_start_window,
        finetune_fused_backward=args.finetune_fused_backward,
        finetune_scope=args.finetune_scope,
        finetune_blocks=args.finetune_blocks,
        finetune_master=args.finetune_master,
        finetune_scratch_dir=args.finetune_scratch_dir,
        reg_lr_multiplier=args.reg_lr_multiplier,
    )


if __name__ == "__main__":
    main()
