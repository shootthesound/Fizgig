"""Krea 2 LoRA training CLI (wraps fizgig.krea2.trainer.train_krea2).

The GUI drives the full run as three steps: krea2_cache_latents -> krea2_cache_text -> krea2_train.
Trains on the RAW model; previews render on the fp8 Turbo (--turbo_dit) with the live LoRA.
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# CUDA allocator policy, set before torch is imported below — the backend is fixed at CUDA init.
# Training churns the allocator: large tensors allocated and freed every step, and on a rotating
# fine-tune whole windows swap between bf16 and fp8 each epoch. The default allocator carves from
# fixed-size segments, which fragments under that pattern and worsens as a run goes on.
# The GUI already sets this and the training subprocess inherits it; this covers headless runs.
# Respects an existing value, and FIZGIG_NO_EXPANDABLE=1 opts out for A/B testing.
if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF") and os.environ.get("FIZGIG_NO_EXPANDABLE") != "1":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# OpenMP wait policy, before torch loads libiomp: Intel OpenMP keeps every pool thread
# actively spinning for 200 ms after each parallel region, so the small per-step CPU ops
# re-arm an all-core busy-spin for the whole run (issue #18 — 100% CPU on every core while
# the actual training is on the GPU). BLOCKTIME=0 measured: 14.8 spinning cores -> 0, no
# step-time cost. The GUI sets this too; this covers headless runs. setdefault, so an
# explicit user value wins.
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

from fizgig.krea2.trainer import train_krea2
from fizgig.training.optimizers import available_optimizers

logging.basicConfig(level=logging.INFO)


def setup_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Krea 2 LoRA training (RAW base, fp8 Turbo previews)")
    p.add_argument("--dit", required=True, help="Krea 2 RAW DiT (Krea-2-raw.safetensors)")
    p.add_argument("--dataset_config", required=True, help="Dataset .toml")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_name", required=True)
    p.add_argument("--network_dim", type=int, default=32)
    p.add_argument("--network_alpha", type=float, default=32)
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
                   help="Keep only the N newest state dirs (each is LoRA + optimizer, hundreds "
                        "of MB). Clamped to >= 1.")
    p.add_argument("--no_fp8", action="store_true", help="Train the base in bf16 instead of dynamic fp8")
    p.add_argument("--quantize_4bit", action="store_true",
                   help="QLoRA-style 4-bit (NF4) frozen base — ~5.6 GB DiT, fits 10-12 GB cards (no block swap)")
    p.add_argument("--quant_int8", default="", choices=["", "bf16", "int8"],
                   help="EXPERIMENTAL INT8 W8A8 frozen base. 'bf16' = int8 forward with exact "
                        "bf16 gradients; 'int8' = both quantised (faster, lossier)")
    p.add_argument("--blocks_to_swap", type=int, default=0)
    p.add_argument("--discrete_flow_shift", type=float, default=2.5)
    p.add_argument("--seed", type=int, default=42)
    # previews (sample the fp8 Turbo with the live LoRA)
    p.add_argument("--turbo_dit", default=None, help="Pre-quant fp8 Turbo for previews")
    p.add_argument("--turbo_lora", default=None,
                   help="Turbo distillation LoRA (rank 64) — previews render on the resident "
                        "training DiT with this at strength 1.0 instead of loading the Turbo "
                        "checkpoint (no CPU parking; same 8-step CFG-free settings)")
    p.add_argument("--vae", default=None, help="Qwen-Image VAE (for preview decode)")
    p.add_argument("--text_encoder", default=None, help="bf16 Qwen3-VL-4B (for preview prompt encode)")
    p.add_argument("--sample_prompts", default=None, help="Sample-prompts file (one prompt per line)")
    p.add_argument("--sample_every_n_epochs", type=int, default=0)
    p.add_argument("--sample_width", type=int, default=512)
    p.add_argument("--sample_height", type=int, default=512)
    p.add_argument("--sample_steps", type=int, default=8, help="Preview denoising steps (Turbo default 8)")
    p.add_argument("--sample_cfg_scale", type=float, default=1.0,
                   help=">1 enables CFG on the Turbo previews (pair with --sample_negative)")
    p.add_argument("--sample_negative", default=None,
                   help="Negative prompt for previews — only used when --sample_cfg_scale > 1")
    p.add_argument("--sample_at_first", action="store_true",
                   help="Render an epoch-0 preview before training starts")
    p.add_argument("--sample_seed", type=int, default=42, help="Seed for in-training preview samples")
    p.add_argument("--sample_ref_image", default=None, help="Reference image (Qwen3-VL vision path)")
    # Output metadata (recorded in the saved LoRA)
    p.add_argument("--metadata_title", default=None)
    p.add_argument("--metadata_author", default=None)
    p.add_argument("--metadata_description", default=None)
    p.add_argument("--metadata_license", default=None)
    p.add_argument("--metadata_tags", default=None)
    p.add_argument("--metadata_trigger_phrase", default=None,
                   help="Trigger word(s) recorded as modelspec.trigger_phrase. "
                        "Defaults to --trigger_word when omitted.")
    p.add_argument("--metadata_thumbnail", default=None,
                   help="Path to an image to embed as modelspec.thumbnail. Omit to "
                        "auto-use the latest sample preview; pass 'off' to disable.")
    p.add_argument("--preview_blocks_to_swap", type=int, default=0,
                   help="Forward-only block swap on the preview Turbo (fits smaller cards)")
    p.add_argument("--preview_int8", action="store_true",
                   help="INT8 (W8A8) fast matmul for the preview Turbo (experimental, same VRAM as fp8)")
    p.add_argument("--resume", default=None, help="Path to a <name>-NNNNNN-state dir to resume from")
    p.add_argument("--context_lora_path", default=None, help="Frozen+active context LoRA on the base")
    p.add_argument("--context_lora_strength", type=float, default=1.0)
    p.add_argument("--adaptive_lr", action="store_true", help="Bi-directional plateau LR tracker")
    p.add_argument("--adaptive_lr_min", type=float, default=1e-5)
    p.add_argument("--adaptive_lr_max", type=float, default=4e-4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="Accumulate grads over N micro-batches per optimizer step (effective batch = N)")
    p.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm (0 disables)")
    p.add_argument("--optimizer_type", default="adamw8bit",
                   help="Optimizer family, or a full module.path.ClassName. Available here: "
                        + ", ".join(available_optimizers()))
    p.add_argument("--optimizer_args", default="",
                   help='Extra optimizer kwargs, e.g. "weight_decay=0.01 betas=0.9,0.99"')
    p.add_argument("--compile_blocks", default="auto", choices=["auto", "on", "off"],
                   help="torch.compile the transformer blocks. 'auto' (default) enables it only "
                        "when the run is long enough to repay its ~90 s warm-up and the VRAM fits "
                        "— roughly 600+ steps on INT8, 1200+ on NF4. Measured 2.0x per step on "
                        "INT8 (0.59 -> 0.29) and 1.28x on NF4 (0.71 -> 0.56). Needs triton and, on "
                        "Windows, MSVC; never used under block swap")
    p.add_argument("--lr_scheduler", default="constant",
                   choices=["constant", "constant_with_warmup", "cosine", "cosine_with_restarts",
                            "linear", "polynomial"],
                   help="Step-level LR schedule. Ignored when --adaptive_lr is set (that watcher owns the LR)")
    p.add_argument("--lr_warmup_steps", type=int, default=0, help="Warmup steps for the LR scheduler")
    p.add_argument("--lr_decay_steps", type=int, default=0, help="Reserved for parity with Klein (unused)")
    p.add_argument("--lr_scheduler_num_cycles", type=int, default=1, help="Cycles for cosine_with_restarts")
    p.add_argument("--lr_scheduler_power", type=float, default=1.0, help="Power for the polynomial schedule")
    p.add_argument("--log_per_image_loss", action="store_true",
                   help="Per-image loss tracking + stuck-image detection (loss_log/problem_images.json)")
    p.add_argument("--per_image_lr", action="store_true",
                   help="Per-image adaptive LR: throttle stuck images, boost healthy learned ones (experimental)")
    p.add_argument("--auto_recaption", action="store_true",
                   help="Auto-recaption confirmed-stuck images with Qwen3-VL between epochs (experimental)")
    p.add_argument("--warmup_look_outliers", action="store_true",
                   help="LR warm-up (x0.4->x1.0 over first epochs) for Look Filter outlier images "
                        "(reads <dataset>/fizgig_look_scores.json from the Image Prep Look Filter)")
    p.add_argument("--trigger_word", default=None,
                   help="Trigger word appended (', <trigger>') to auto-generated captions")
    p.add_argument("--recaption_instruction", default=None,
                   help="Instruction for auto-recaption attempt 1 (the Captions tab's edited "
                        "Training-caption preset). Omit to use the built-in")
    p.add_argument("--recaption_instruction_detailed", default=None,
                   help="Instruction for auto-recaption attempt 2 (the Captions tab's edited "
                        "Exhaustive-detail preset). Omit to use the built-in")
    return p


def main():
    args = setup_parser().parse_args()
    prompts = None
    if args.sample_prompts and os.path.exists(args.sample_prompts):
        with open(args.sample_prompts, encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    # Reference-only samples: a reference image with no prompt is a valid 'generate from this
    # picture' run (Qwen3-VL vision path) — render one empty-prompt sample driven by the ref.
    if not prompts and args.sample_ref_image:
        prompts = [""]

    train_krea2(
        args.dit, args.dataset_config, args.output_dir, args.output_name,
        network_dim=args.network_dim, network_alpha=args.network_alpha,
        network_type=args.network_type, lokr_factor=args.lokr_factor,
        learning_rate=args.learning_rate, max_train_epochs=args.max_train_epochs,
        save_every_n_epochs=args.save_every_n_epochs, fp8_scaled=not args.no_fp8,
        save_state=args.save_state, save_state_on_train_end=args.save_state_on_train_end,
        keep_last_n_states=args.keep_last_n_states,
        quant_4bit=args.quantize_4bit, quant_int8=args.quant_int8,
        blocks_to_swap=args.blocks_to_swap, shift=args.discrete_flow_shift, seed=args.seed,
        sample_prompts=prompts, turbo_path=args.turbo_dit, turbo_lora_path=args.turbo_lora,
        vae_path=args.vae, te_path=args.text_encoder,
        sample_every_n_epochs=args.sample_every_n_epochs,
        sample_width=args.sample_width, sample_height=args.sample_height,
        sample_steps=args.sample_steps, sample_cfg_scale=args.sample_cfg_scale,
        sample_negative=args.sample_negative, sample_at_first=args.sample_at_first,
        sample_seed=args.sample_seed,
        sample_ref_image=args.sample_ref_image,
        metadata_title=args.metadata_title, metadata_author=args.metadata_author,
        metadata_description=args.metadata_description,
        metadata_license=args.metadata_license, metadata_tags=args.metadata_tags,
        metadata_trigger_phrase=args.metadata_trigger_phrase or args.trigger_word,
        metadata_thumbnail=args.metadata_thumbnail,
        preview_blocks_to_swap=args.preview_blocks_to_swap,
        preview_int8=args.preview_int8,
        resume_state_dir=args.resume,
        context_lora_path=args.context_lora_path, context_lora_strength=args.context_lora_strength,
        adaptive_lr=args.adaptive_lr,
        adaptive_lr_min=args.adaptive_lr_min, adaptive_lr_max=args.adaptive_lr_max,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        optimizer_type=args.optimizer_type, optimizer_args=args.optimizer_args,
        compile_blocks=args.compile_blocks,
        lr_scheduler=args.lr_scheduler, lr_warmup_steps=args.lr_warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        lr_scheduler_num_cycles=args.lr_scheduler_num_cycles,
        lr_scheduler_power=args.lr_scheduler_power,
        log_per_image_loss=args.log_per_image_loss,
        per_image_lr=args.per_image_lr,
        auto_recaption=args.auto_recaption,
        warmup_look_outliers=args.warmup_look_outliers,
        trigger_word=args.trigger_word,
        recaption_instruction=args.recaption_instruction,
        recaption_instruction_detailed=args.recaption_instruction_detailed,
    )


if __name__ == "__main__":
    main()
