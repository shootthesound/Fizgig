"""Standard-layer caching for any described family: VAE latents or text conditioning, through the family driver.

    python src/fizgig/families/cache.py --family qwen_image21 --stage latents --dataset_config X.toml --model VAE
    python src/fizgig/families/cache.py --family qwen_image21 --stage text    --dataset_config X.toml --model TE

Uses Fizgig's dataset framework (bucketing, stale-cache cleanup, --skip_existing) exactly like the per-family
scripts. Latents are stored as `latent_{h}x{w}`; conditioning as `cond__<driver key>` (passed through verbatim by
the dataset loader and handed back to the driver as the same dict).

Edit pairs (a dataset with `control_directory`, for a driver with supports_references; up to the description's
edit_references before-images per after-image): the before-images are cached
beside each target as `latent_control_{i}_{h}x{w}`, the first at the target's bucket and extra references at their own
shape and the target's pixel count, and the text stage encodes every caption
WITH its before-images at that same size (the text encoder sees the references, and its image tokens must line up
with their latents). The text cache records that size, so a changed Target Megapixels re-encodes it.
"""
import argparse
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from safetensors.torch import save_file  # noqa: E402

from fizgig.dataset.config import (BlueprintGenerator, ConfigSanitizer,  # noqa: E402
                                   generate_dataset_group_by_blueprint, load_user_config)
from fizgig.dataset.image_dataset import dtype_to_str, source_stamp  # noqa: E402
from fizgig.families.registry import get  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
FORMAT_VERSION = "1.0.0"


def _clean(t, what, key):
    t = t.detach().cpu().contiguous()
    if t.is_floating_point() and torch.isnan(t).any():
        logger.warning(f"NaN in {what} for {key} - replaced with 0")
        t[torch.isnan(t)] = 0
    return t


def _latent_key(latent):
    """A still (C, h, w) -> latent_{h}x{w}; a clip (C, T, h, w) -> latent_{T}x{h}x{w} (H3's rule - both start
    `latent_`, which is all the dataset collate looks at)."""
    return "latent_" + "x".join(str(d) for d in latent.shape[1:])


def save_latents(desc, item, latent, controls=(), extra=None):
    """extra: further tensors the driver stores beside the latent (e.g. a clip's audio rows), keyed as given."""
    os.makedirs(os.path.dirname(item.latent_cache_path), exist_ok=True)
    sd = {_latent_key(latent): _clean(latent, "latent", item.item_key)}
    for k, v in (extra or {}).items():
        sd[k] = _clean(v, k, item.item_key)
    for i, c in enumerate(controls):
        sd[f"latent_control_{i}_{c.shape[-2]}x{c.shape[-1]}"] = _clean(c, "control latent", item.item_key)
    save_file(sd, item.latent_cache_path, metadata={
        "architecture": desc.arch_id, "width": str(item.original_size[0]), "height": str(item.original_size[1]),
        "dtype": dtype_to_str(latent.dtype), "format_version": FORMAT_VERSION, "source": source_stamp(item.item_key)})


def _ref_sizes(item):
    """'WxH,WxH' of an item's before-images as cached (the text cache is only valid at these)."""
    return ",".join(f"{c.shape[1]}x{c.shape[0]}" for c in (item.control_content or []))


def save_cond(desc, item, cond, refs=""):
    os.makedirs(os.path.dirname(item.text_encoder_output_cache_path), exist_ok=True)
    md = {"architecture": desc.arch_id, "caption1": item.caption, "format_version": FORMAT_VERSION}
    if refs:
        md["reference_sizes"] = refs
    # a None mask (e.g. Fibo 2.0's encode_prompt) is dropped, matching the reference trainer's pack()
    save_file({f"cond__{k}": _clean(v, k, item.item_key) for k, v in cond.items() if v is not None},
              item.text_encoder_output_cache_path, metadata=md)


def save_cond_file(desc, path, caption, cond, key=""):
    """A text encoding at a path of its own, in the text cache's format (DOP's trigger-free captions sit beside each
    item's own cache)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    md = {"architecture": desc.arch_id, "caption1": caption, "format_version": FORMAT_VERSION}
    save_file({f"cond__{k}": _clean(v, k, key) for k, v in cond.items()}, path, metadata=md)


def cond_file_matches(path, caption):
    """An existing text encoding written from this caption."""
    return _cached_matches(path, caption, "")


def _cached_matches(path, caption, refs):
    """An existing text cache that still fits: same caption and same before-image sizes."""
    from safetensors import safe_open
    try:
        with safe_open(path, framework="pt") as f:
            md = f.metadata() or {}
    except Exception:
        return False
    return md.get("caption1") == caption and md.get("reference_sizes", "") == refs


def _has_controls(path):
    from safetensors import safe_open
    try:
        with safe_open(path, framework="pt") as f:
            return any(k.startswith("latent_control_") for k in f.keys())
    except Exception:
        return False


def _encode_text_with_references(args, datasets, driver, te, desc):
    """The text stage for edit pairs: walks the latent batches (they carry the bucket-sized before-images)."""
    from fizgig.scripts.cache_text import post_process, prepare_cache_files_and_paths
    files, paths = prepare_cache_files_and_paths(datasets)
    workers = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() - 1)
    for i, ds in enumerate(datasets):
        logger.info(f"Encoding dataset [{i}] with its before-images")
        for _, batch in ds.retrieve_latent_cache_batches(workers):
            for it in batch:
                p, refs = os.path.normpath(it.text_encoder_output_cache_path), _ref_sizes(it)
                paths[i].add(p)
                if args.skip_existing and p in files[i] and _cached_matches(p, it.caption, refs):
                    continue
                c = driver.encode_text_with_references(te, [it.caption], [it.control_content])[0]
                logger.info(f"text cache: {it.item_key} with {len(it.control_content)} before-image(s) at {refs}")
                save_cond(desc, it, c, refs)
    post_process(datasets, files, paths, args.keep_cache)


def main():
    p = argparse.ArgumentParser(description="Cache latents or text conditioning for a described model family")
    p.add_argument("--family", required=True, help="family key, e.g. qwen_image21")
    p.add_argument("--stage", required=True, choices=["latents", "text"])
    p.add_argument("--dataset_config", required=True)
    p.add_argument("--model", required=True, help="the VAE (latents) or text encoder (text) file")
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--keep_cache", action="store_true")
    p.add_argument("--slider", action="store_true",
                   help="the control_directory holds a slider's other pole: cache its latents, encode captions "
                        "plainly (not as edit pairs)")
    p.add_argument("--aux", action="append", default=[], metavar="KEY=VALUE",
                   help="a further model file or option for the family's own caching (e.g. audio_vae=PATH)")
    args = p.parse_args()
    from fizgig.families.quant import apply_vram_cap
    apply_vram_cap()                # FIZGIG_SIM_VRAM_GB: behave like a smaller card

    desc = get(args.family)
    if desc is None or not desc.training_ready:
        raise SystemExit(f"unknown or untrainable family {args.family!r}")
    driver = desc.load_driver()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(load_user_config(args.dataset_config), args,
                                                               architecture=desc.arch_id)
    datasets = generate_dataset_group_by_blueprint(blueprint.dataset_group).datasets
    pairs = any(getattr(ds, "has_control", False) for ds in datasets)
    if pairs and args.slider and not desc.slider_training:
        raise SystemExit(f"{desc.display_name} has no slider training")
    if pairs and not args.slider and not driver.supports_references:
        raise SystemExit(f"{desc.display_name} has no edit training: remove control_directory from the dataset")
    limit = 1 if args.slider else max(1, getattr(desc, "edit_references", 1))
    for ds in datasets:        # a slider pairs one photo with one; an edit takes up to the family's references
        many = [os.path.basename(p) for p, m in getattr(ds.datasource, "control_paths", {}).items() if len(m) > limit]
        if many:
            raise SystemExit(f"{len(many)} after-image(s) match more than {limit} before-image(s) (e.g. "
                             f"{', '.join(many[:3])}): {desc.display_name} takes "
                             + ("one before-image per after-image, named the same" if limit == 1 else
                                f"up to {limit} (photo.png, then photo_1.png, photo_2.png ...)"))

    aux = dict(a.split("=", 1) for a in args.aux if "=" in a)
    if driver.cache_stage(args.stage, datasets, args, device, aux):
        return                              # the family caches in its own layout (H3)
    if args.stage == "latents":
        from fizgig.scripts.cache_latents import encode_datasets
        vae = driver.load_vae(args.model, device)
        # a cache from before the pairs were added (or after they were removed) no longer fits: re-encode it
        args.needs_reencode = lambda path: _has_controls(path) != pairs

        def encode(batch):
            imgs = [(it.content[0] if isinstance(it.content, list) else it.content) for it in batch]
            for it, z in zip(batch, driver.encode_images(vae, imgs)):
                # one at a time: a multi-reference edit's extra pictures have their own shapes, and a driver may
                # stack its batch (Klein)
                ctrl = [driver.encode_images(vae, [c])[0] for c in (it.control_content or [])]
                logger.info(f"latent cache: {it.item_key} -> {tuple(z.shape)}"
                            + (f" + {len(ctrl)} before-image(s)" if ctrl else ""))
                save_latents(desc, it, z, ctrl)
        encode_datasets(datasets, encode, args)
    else:
        from fizgig.scripts.cache_text import post_process, prepare_cache_files_and_paths, process_batches
        if pairs and not args.slider:
            te = driver.load_reference_text_encoder(args.model, device)
            _encode_text_with_references(args, datasets, driver, te, desc)
            driver.unload_text_encoder(te)
            return
        if args.batch_size is None:
            args.batch_size = 8
        files, paths = prepare_cache_files_and_paths(datasets)
        te = driver.load_text_encoder(args.model, device)

        def encode(batch):
            for it, c in zip(batch, driver.encode_text(te, [it.caption for it in batch])):
                save_cond(desc, it, c)
        process_batches(args, datasets, files, paths, encode)
        driver.unload_text_encoder(te)
        post_process(datasets, files, paths, args.keep_cache)


if __name__ == "__main__":
    main()
