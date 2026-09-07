"""Clip stills as photo items (MiniMax H3): the sharpest-face picker, the cache keys, the
derived dataset items, the frame-0 fallback, no sound, stills and voice untouched, off by
default. Headless, CPU; uses one real clip from Peter's videotest set for the picker when it
is there (S:/dreambooth/sydney/videotest), else skips that section.

Run: venv/Scripts/python.exe tests/test_h3_clip_still.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from safetensors.torch import save_file

from fizgig.dataset.image_dataset import ImageDataset, BucketSelector, BUCKET_RESO_STEPS
from fizgig.training.metadata import ARCHITECTURE_MINIMAX
from fizgig.minimax.still_pick import pick_still_frame

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


# ---- the picker -----------------------------------------------------------------------------
REAL = r"S:/dreambooth/sydney/videotest"
clips = sorted(f for f in os.listdir(REAL) if f.endswith(".mp4")) if os.path.isdir(REAL) else []
if clips:
    import cv2
    from PIL import Image
    from fizgig.minimax.clip import read_frames
    fr = read_frames(os.path.join(REAL, clips[0]))
    s = 512 / min(fr.shape[1], fr.shape[2])
    fr = np.stack([np.asarray(Image.fromarray(f).resize((round(fr.shape[2] * s), round(fr.shape[1] * s)))) for f in fr])
    idx, info = pick_still_frame(fr)
    ck("picker: a real clip yields a frame with a face", info["face"] and 0 <= idx < len(fr), (idx, info))
    # blur every frame but two: the pick must be one of the sharp ones
    blurred = fr.copy()
    keep = {3, 11} if len(fr) > 11 else {0, 1}
    for i in range(len(blurred)):
        if i not in keep:
            blurred[i] = cv2.GaussianBlur(blurred[i], (0, 0), 6)
    idx2, info2 = pick_still_frame(blurred)
    ck("picker: with all but two frames blurred it picks a sharp one", idx2 in keep and info2["face"], (idx2, info2))
    ck("picker: a one-frame input is frame 0, no detection", pick_still_frame(fr[:1]) == (0, {"face": False, "score": 0.0, "detections": 0, "candidates": 0}))
else:
    print("(videotest not present — picker section skipped)")
noise = (np.random.RandomState(0).rand(6, 96, 96, 3) * 255).astype(np.uint8)
idx3, info3 = pick_still_frame(noise)
ck("picker: no face anywhere -> frame 0, flagged", idx3 == 0 and info3["face"] is False and info3["detections"] == 6, info3)

# ---- the dataset ---------------------------------------------------------------------------
TMP = tempfile.mkdtemp(prefix="clipstill_")
IMG, CACHE = TMP, os.path.join(TMP, "cache"); os.makedirs(CACHE)
sel = BucketSelector((496, 496), True, True, BUCKET_RESO_STEPS[ARCHITECTURE_MINIMAX])
bw, bh = sel.get_bucket_resolution((832, 576)); lh, lw = bh // 16, bw // 16
STILL = torch.randn(24, lh, lw)
for stem, shape in (("clip_picked", (24, 7, lh, lw)), ("clip_plain", (24, 7, lh, lw)), ("photo", (24, lh, lw))):
    key = "latent_" + "x".join(str(v) for v in shape[1:])
    sd = {key: torch.randn(shape)}
    if len(shape) == 4:
        sd["audio_latent"] = torch.randn(8, 32)
    if stem == "clip_picked":
        sd["still_latent"] = STILL
        sd["still_frame"] = torch.tensor(9)
    save_file(sd, os.path.join(CACHE, f"{stem}_0832x0576_{ARCHITECTURE_MINIMAX}.safetensors"))
    save_file({"hidden_states": torch.zeros(8, 5120), "attention_mask": torch.ones(8, dtype=torch.bool)},
              os.path.join(CACHE, f"{stem}_{ARCHITECTURE_MINIMAX}_te.safetensors"))
    open(os.path.join(TMP, f"{stem}.mp4" if stem.startswith("clip") else f"{stem}.png"), "wb").close()
    open(os.path.join(TMP, f"{stem}.txt"), "w").write("a caption")

ck("header peek: has_still true only for the picked clip",
   ImageDataset.latent_cache_has_still(os.path.join(CACHE, f"clip_picked_0832x0576_{ARCHITECTURE_MINIMAX}.safetensors"))
   and not ImageDataset.latent_cache_has_still(os.path.join(CACHE, f"clip_plain_0832x0576_{ARCHITECTURE_MINIMAX}.safetensors")))


def make(flag):
    ImageDataset.clip_still_as_photo = flag
    ds = ImageDataset(image_directory=IMG, caption_extension=".txt", resolution=(496, 496),
                      architecture=ARCHITECTURE_MINIMAX, batch_size=1, num_repeats=1,
                      enable_bucket=True, bucket_no_upscale=True, cache_directory=CACHE)
    ds.prepare_for_training()
    return ds


def batch_for(ds, key):
    i = next(i for i, (r, b) in enumerate(ds.batch_manager.bucket_batch_indices)
             if ds.batch_manager.buckets[r][b].item_key == key)
    return ds.batch_manager[i]


try:
    off = make(False)
    keys_off = [b.item_key for bk in off.batch_manager.buckets.values() for b in bk]
    ck("off by default: no derived items", len(off) == 3 and not any("#still" in k for k in keys_off), keys_off)

    on = make(True)
    keys_on = sorted(b.item_key for bk in on.batch_manager.buckets.values() for b in bk)
    ck("on: one still item per clip, photo untouched",
       len(on) == 5 and sum(k.endswith("#still") for k in keys_on) == 2, keys_on)
    picked_key = next(k for k in keys_on if k.endswith("#still") and "clip_picked" in k)
    plain_key = next(k for k in keys_on if k.endswith("#still") and "clip_plain" in k)
    b_p = batch_for(on, picked_key)
    b_c = batch_for(on, picked_key[:-len("#still")])
    ck("a picked clip's still item trains the cached still_latent",
       b_p["latents"].dim() == 4 and torch.equal(b_p["latents"][0], STILL), tuple(b_p["latents"].shape))
    ck("...with no sound and no still keys of its own",
       "audio_latent" not in b_p and not any("still" in k for k in b_p))
    ck("the clip's own batch carries neither still key",
       b_c["latents"].dim() == 5 and "audio_latent" in b_c and not any(k.startswith("still") for k in b_c), sorted(b_c))
    b_q = batch_for(on, plain_key)
    b_qc = batch_for(on, plain_key[:-len("#still")])
    ck("a clip cached without a pick falls back to frame 0 of its latent",
       b_q["latents"].dim() == 4 and torch.equal(b_q["latents"][0], b_qc["latents"][0, :, 0]))
    ck("the item key names the still for the loss logger", b_p["item_keys"] == [picked_key])
finally:
    ImageDataset.clip_still_as_photo = False

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("ALL PASS")
