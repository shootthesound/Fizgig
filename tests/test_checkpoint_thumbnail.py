"""#122 — an epoch checkpoint's auto thumbnail was the PREVIOUS epoch's preview, because the
checkpoint is saved before its own preview renders. Pins: sample_for_epoch picks by epoch
number (not mtime), refresh_checkpoint_thumbnail swaps only the thumbnail and leaves every
tensor byte-identical, and both the Klein and Krea 2 trainers call the refresh after the
epoch preview (and never when an explicit --metadata_thumbnail was given). No GPU.

Run: venv/Scripts/python.exe tests/test_checkpoint_thumbnail.py
"""
import io as _io
import os
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file, load_file  # noqa: E402

from fizgig.training.metadata import (sample_for_epoch, refresh_checkpoint_thumbnail,  # noqa: E402
                                      latest_sample_image, thumbnail_data_uri)

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


d = tempfile.mkdtemp(prefix="thumb122_")
sd = os.path.join(d, "sample"); os.makedirs(sd)


def png(name, colour):
    p = os.path.join(sd, name); Image.new("RGB", (32, 32), colour).save(p); return p


# another run's file first, then previews 14, 15, 16 — 16 is newest on disk
png("other_e000015_00_20260909100300_1.png", (9, 9, 9)); time.sleep(0.02)
p14 = png("me_e000014_00_20260909100000_1.png", (255, 0, 0)); time.sleep(0.02)
p15 = png("me_e000015_00_20260909100100_1.png", (0, 255, 0)); time.sleep(0.02)
p16 = png("me_e000016_00_20260909100200_1.png", (0, 0, 255))

ck("latest_sample_image is mtime-newest (the old behaviour: epoch 16)", latest_sample_image(d) == p16)
ck("sample_for_epoch(15) picks epoch 15 by name, not the newest file", sample_for_epoch(d, "me", 15) == p15)
ck("...and ignores another run's epoch-15 file", "other_" not in os.path.basename(sample_for_epoch(d, "me", 15)))
ck("no preview for that epoch -> None", sample_for_epoch(d, "me", 99) is None)

# a checkpoint saved with the stale (epoch-14) thumbnail, then refreshed with its own
ck_path = os.path.join(d, "me-000015.safetensors")
tensors = {"lora_a": torch.randn(4, 4), "lora_b": torch.randn(2, 4), "alpha": torch.tensor(4.0)}
save_file(tensors, ck_path, metadata={"modelspec.thumbnail": thumbnail_data_uri(p14), "modelspec.title": "me", "ss_epoch": "15"})
ok = refresh_checkpoint_thumbnail(ck_path, sample_for_epoch(d, "me", 15))
ck("refresh returns True", ok)
with safe_open(ck_path, "pt") as f:
    meta = f.metadata()
ck("thumbnail now equals epoch 15's preview", meta["modelspec.thumbnail"] == thumbnail_data_uri(p15))
ck("other metadata kept", meta["modelspec.title"] == "me" and meta["ss_epoch"] == "15")
after = load_file(ck_path)
ck("every tensor byte-identical after the refresh",
   set(after) == set(tensors) and all(torch.equal(after[k], tensors[k]) for k in tensors))
ck("no temp file left behind", not os.path.exists(ck_path + ".thumb.tmp"))
ck("a second refresh with the same image is a no-op that still returns True",
   refresh_checkpoint_thumbnail(ck_path, p15) and os.path.getmtime(ck_path) > 0)
ck("missing checkpoint -> False, no exception", refresh_checkpoint_thumbnail(os.path.join(d, "nope.safetensors"), p15) is False)

# --- both trainers wire it after the epoch preview, gated on an AUTO thumbnail ---------------
klein = _io.open(os.path.join(REPO, "src", "fizgig", "training", "trainer.py"), encoding="utf-8").read()
krea2 = _io.open(os.path.join(REPO, "src", "fizgig", "krea2", "trainer.py"), encoding="utf-8").read()
for name, src, gate in (("Klein", klein, 'not (args.metadata_thumbnail or "").strip()'),
                        ("Krea 2", krea2, 'not (metadata_thumbnail or "").strip()')):
    i_prev = src.index("sample_images(accelerator, args, epoch + 1" if name == "Klein" else "_, _last_p = sample_previews(turbo_path, sample_ae, prev_enc")
    i_ref = src.index("refresh_checkpoint_thumbnail(", i_prev)
    ck(f"{name}: refreshes the epoch checkpoint AFTER the epoch preview", i_ref > i_prev and i_ref - i_prev < 1500)
    ck(f"{name}: only when the thumbnail is automatic (explicit --metadata_thumbnail stays)", gate in src[i_prev:i_ref])
    ck(f"{name}: picks the preview by epoch number", "sample_for_epoch(" in src[i_prev:i_ref + 200])

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}"); sys.exit(1)
print("ALL PASS")
