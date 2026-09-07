"""Every clip's first frame as a photo item (MiniMax H3): derived items, the slice, no
sound, stills and voice untouched, off by default. Headless — reads Peter's videotest cache
when it is there (S:/dreambooth/sydney/videotest), else a synthetic cache.

Run: venv/Scripts/python.exe tests/test_h3_clip_first_frame.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from safetensors.torch import save_file

from fizgig.dataset.image_dataset import ImageDataset, BucketSelector, BUCKET_RESO_STEPS
from fizgig.training.metadata import ARCHITECTURE_MINIMAX

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


REAL = r"S:/dreambooth/sydney/videotest"
REAL_CACHE = r"W:/Peter/Documents/Development/Fizgig/cache/videotest-6fecac16"
if os.path.isdir(REAL) and os.path.isdir(REAL_CACHE):
    IMG, CACHE = REAL, REAL_CACHE
    print("using the real videotest dataset")
else:
    TMP = tempfile.mkdtemp(prefix="firstframe_")
    IMG, CACHE = TMP, os.path.join(TMP, "cache"); os.makedirs(CACHE)
    sel = BucketSelector((496, 496), True, True, BUCKET_RESO_STEPS[ARCHITECTURE_MINIMAX])
    bw, bh = sel.get_bucket_resolution((832, 576)); lh, lw = bh // 16, bw // 16
    for stem, shape in (("clip_1", (24, 7, lh, lw)), ("clip_2", (24, 7, lh, lw)), ("photo", (24, lh, lw))):
        key = "latent_" + "x".join(str(v) for v in shape[1:])
        t = torch.randn(shape)
        sd = {key: t}
        if len(shape) == 4:
            sd["audio_latent"] = torch.randn(1, 8, 32)
        save_file(sd, os.path.join(CACHE, f"{stem}_0832x0576_{ARCHITECTURE_MINIMAX}.safetensors"))
        save_file({"hidden_state": torch.zeros(1, 8, 5120)}, os.path.join(CACHE, f"{stem}_{ARCHITECTURE_MINIMAX}_te.safetensors"))
        open(os.path.join(TMP, f"{stem}.mp4" if stem.startswith("clip") else f"{stem}.png"), "wb").close()
        open(os.path.join(TMP, f"{stem}.txt"), "w").write("a caption")
    print("using a synthetic cache")


def make(flag):
    ImageDataset.clip_first_frame_as_photo = flag
    ds = ImageDataset(image_directory=IMG, caption_extension=".txt", resolution=(496, 496),
                      architecture=ARCHITECTURE_MINIMAX, batch_size=1, num_repeats=1,
                      enable_bucket=True, bucket_no_upscale=True, cache_directory=CACHE)
    ds.prepare_for_training()
    return ds


try:
    off = make(False)
    n_off = len(off)
    keys_off = [b.item_key for bk in off.batch_manager.buckets.values() for b in bk]
    n_clips = sum(1 for bk in off.batch_manager.buckets.values() for b in bk
                  if ImageDataset.latent_cache_frames(b.latent_cache_path) > 1)
    ck("off by default: no derived items", not any("#frame0" in k for k in keys_off) and n_clips > 0, (n_off, n_clips))

    on = make(True)
    keys_on = [b.item_key for bk in on.batch_manager.buckets.values() for b in bk]
    ck("on: one first-frame item per clip, stills untouched",
       len(on) == n_off + n_clips and sum(1 for k in keys_on if k.endswith("#frame0")) == n_clips,
       (len(on), n_off, n_clips))
    # find a derived item and its parent clip, load both through the real __getitem__
    idx_ff = next(i for i, (reso, bi) in enumerate(on.batch_manager.bucket_batch_indices)
                  if on.batch_manager.buckets[reso][bi].item_key.endswith("#frame0"))
    reso, bi = on.batch_manager.bucket_batch_indices[idx_ff]
    ff_item = on.batch_manager.buckets[reso][bi]
    parent_key = ff_item.item_key[:-len("#frame0")]
    idx_clip = next(i for i, (r, b) in enumerate(on.batch_manager.bucket_batch_indices)
                    if on.batch_manager.buckets[r][b].item_key == parent_key)
    b_ff = on.batch_manager[idx_ff]
    b_clip = on.batch_manager[idx_clip]
    ck("the derived batch is a still: latents [1, 24, H, W], equal to the clip's first latent frame",
       b_ff["latents"].dim() == 4 and b_clip["latents"].dim() == 5
       and torch.equal(b_ff["latents"][0], b_clip["latents"][0, :, 0]),
       (tuple(b_ff["latents"].shape), tuple(b_clip["latents"].shape)))
    ck("...with no sound and the clip's own text encoding",
       "audio_latent" not in b_ff and "audio_only" not in b_ff
       and all(torch.equal(b_ff[k], b_clip[k]) for k in b_clip if k not in ("latents", "audio_latent", "audio_only", "item_keys", "timesteps") and torch.is_tensor(b_clip[k])),
       sorted(b_ff.keys()))
    ck("the item key names the frame for the loss logger", b_ff["item_keys"] == [parent_key + "#frame0"])
    ck("header peek: a clip reports its frame count, a still 1",
       ImageDataset.latent_cache_frames(ff_item.latent_cache_path) > 1)
finally:
    ImageDataset.clip_first_frame_as_photo = False

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("ALL PASS")
