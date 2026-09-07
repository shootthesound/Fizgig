"""Cache MiniMax H3 video-VAE latents (24-ch) for image-only training.

    python src/fizgig/scripts/minimax_cache_latents.py --dataset_config config.toml --vae path/to/minimax_h3_video_vae
"""

import argparse
import logging
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fizgig.dataset.config import (
    BlueprintGenerator,
    ConfigSanitizer,
    generate_dataset_group_by_blueprint,
    load_user_config,
)
from fizgig.scripts.cache_latents import encode_datasets  # generic dataset iteration / stale-cleanup
from fizgig.minimax.vae import MiniMaxH3VideoVAEEncoder
from fizgig.minimax.caching import encode_and_save_latents
from fizgig.minimax.clip import is_video
from fizgig.training.metadata import ARCHITECTURE_MINIMAX

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache MiniMax H3 video-VAE latents for image training")
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to dataset config .toml file")
    parser.add_argument("--vae", type=str, required=True, help="Path to the H3 video VAE (minimax_h3_video_vae.safetensors)")
    parser.add_argument("--audio_vae", type=str, default=None,
                        help="Path to the H3 audio VAE (minimax_h3_audio_vae_fp32.safetensors). "
                             "Used by video clips with sound (their audio becomes a real "
                             "training target instead of silence; unset, clips train video "
                             "only) and REQUIRED by voice recordings (.wav/.mp3/.flac/.m4a), "
                             "which are nothing but their audio.")
    parser.add_argument("--device", type=str, default=None, help="Device (default: cuda if available)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for encoding")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing cache files")
    parser.add_argument("--keep_cache", action="store_true", help="Keep stale cache files")
    parser.add_argument("--clip_still", action="store_true",
                        help="For every clip, also pick its sharpest frame that shows a face and "
                             "cache it as a still (the 'clip still as a photo' training item). "
                             "With --skip_existing, clips cached without one are re-encoded.")
    return parser


def main():
    from fizgig.utils.device import apply_sim_vram_cap
    apply_sim_vram_cap()          # FIZGIG_SIM_VRAM_GB: behave like a smaller card
    args = setup_parser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    blueprint_gen = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Loading dataset config from {args.dataset_config}")
    user_config = load_user_config(args.dataset_config)
    blueprint = blueprint_gen.generate(user_config, args, architecture=ARCHITECTURE_MINIMAX)
    datasets = generate_dataset_group_by_blueprint(blueprint.dataset_group).datasets

    logger.info(f"Loading H3 video VAE from {args.vae}")
    vae = MiniMaxH3VideoVAEEncoder()
    with safe_open(args.vae, framework="pt", device="cpu") as f:
        vae.load_state_dict({k: f.get_tensor(k) for k in f.keys()}, strict=False)  # decoder keys ignored
    vae = vae.to(device, torch.float32).eval()

    # Loaded only when clips or voice recordings are actually present: it is a 605 MB model that
    # a stills dataset has no use for, and the flag is optional so an existing command line keeps
    # working unchanged.
    from fizgig.minimax.audio import is_audio
    audio_vae = None
    _paths = [p for ds in datasets
              for p in getattr(getattr(ds, "datasource", None), "image_paths", []) or []]
    _has_clips = any(is_video(p) for p in _paths)
    _has_audio = any(is_audio(p) for p in _paths)
    if _has_audio and not args.audio_vae:
        # Clips degrade gracefully without the audio VAE (their video still trains); a voice
        # recording IS its audio, so silence here would cache a placeholder that trains nothing.
        raise ValueError(
            "this dataset contains voice recordings but no audio VAE is configured — set the "
            "Audio VAE path in Preferences (Model Paths, MiniMax H3), or remove the audio files.")
    if (_has_clips or _has_audio) and args.audio_vae:
        from fizgig.minimax.audio_vae import load_minimax_h3_audio_vae
        logger.info(f"Loading H3 audio VAE from {args.audio_vae}")
        audio_vae = load_minimax_h3_audio_vae(args.audio_vae, device=device, dtype=torch.float32)
    elif _has_clips:
        logger.info("[clip] no --audio_vae given: clips will train video only, and any sound "
                    "they carry is ignored.")

    if args.clip_still:
        # --skip_existing keeps a valid clip cache; this hook says a clip cached WITHOUT its
        # still is not yet what this run asked for, so it goes through the encoder again.
        from fizgig.dataset.image_dataset import ImageDataset as _ID
        args.needs_reencode = (lambda path: _ID.latent_cache_frames(path) > 1
                               and not _ID.latent_cache_has_still(path))
    encode_datasets(datasets, lambda batch: encode_and_save_latents(vae, batch, audio_vae,
                                                                    clip_still=args.clip_still), args)


if __name__ == "__main__":
    if sys.platform == "linux" and os.environ.get("FIZGIG_GPU_BACKEND", "").lower() == "rocm":
        from fizgig.rocm.cache_exit import run_cache_main

        run_cache_main(main)
    else:
        main()
