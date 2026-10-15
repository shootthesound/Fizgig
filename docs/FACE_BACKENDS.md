# Face backends: face finder and look match

Fizgig uses faces in two ways:

- the **face finder** (detector) decides where the face is. Image Prep's face close-ups (Resize + face close-ups, Face close-ups only), *Test face detection on one photo* and the clip-still picker use it, and so does the look match when it scores a crop.
- the **look match** (identity embedder) turns a face into a vector, so two faces can be compared. The **Look Consistency Filter**, the **sample gallery's live likeness badges**, **LoRA Royale's likeness score**, **Repair Studio's likeness chips** and the **Block Profiler's likeness/bleed numbers** all use it.

InsightFace and ArcFace are built for human photographs. On a drawn or fursuit character they usually find no face at all (0 of 19 on our test set). For furry and anime datasets, switch both settings.

## Where to set it

**Image Prep tab → 1 · What to do → "Face finder", "Look match" and "YOLO model"**, just under the Face / Padding options. They save as soon as you change them (the YOLO model when you press **Use**), in `user_data/prefs.json` like every other preference, and apply from the next scan or run.

| prefs.json key | Values | Default |
|---|---|---|
| `face_detector` | `auto` (InsightFace **and** the YOLO model on every image, boxes merged), `insightface`, `yolo` | `auto` |
| `face_yolo_model` | path to a YOLO face/head model **.onnx**. A relative path is read relative to the Fizgig folder. Blank means `user_data/models/face/fdetailer_v11.onnx` if that file exists, otherwise the built-in deepghs anime head model. | blank |
| `face_yolo_conf` | the YOLO confidence threshold | `0.35` |
| `face_identity` | `auto` (with the Auto finder: decided per dataset / scoring session, see [Auto look match](#auto-look-match); with `insightface` → ArcFace, with `yolo` → DINOv2), `arcface`, `dinov2`, `clip` | `auto` |
| `face_auto_threshold` | Auto look match: the share of probed images that must have a confident human face for ArcFace | `0.5` |

You can override these for headless runs and tests with `FIZGIG_FACE_DETECTOR`, `FIZGIG_FACE_YOLO_MODEL`, `FIZGIG_FACE_IDENTITY` and `FIZGIG_FACE_AUTO_THRESHOLD`.

**Recommended for every dataset (human, furry, anime or mixed):** finder **Auto**, look match **Auto**, with **Fdetailer v1.1 converted to .onnx** as the YOLO model (see below).

- Auto runs both detectors on every image.
- The look match picks ArcFace for a human dataset and DINOv2 for a furry, anime or mixed one.
- Gender targeting still works on human faces.

Pick **YOLO** or **InsightFace** only to force one detector, for example to save the ~0.13 s per image InsightFace costs on a purely furry set.

## Face finder backends

| Backend | Good at | Gender | Runs on |
|---|---|---|---|
| InsightFace `buffalo_l` (RetinaFace) | human photos | yes | CPU, onnxruntime |
| YOLO model via onnxruntime | drawn, anime and furry characters and fursuits, depending on the model | no | CPU, onnxruntime (no ultralytics, no PyTorch) |
| Auto | everything: **both** on every image, boxes merged | on InsightFace's faces | CPU, ~0.13 s (InsightFace) + ~0.17 s (Fdetailer) per image |

**How Auto merges.**

- A confident InsightFace face (score ≥ 0.6, a real human face) wins over any YOLO box on the same face, and keeps its box and gender. "Same face" means IoU ≥ 0.3, or one box ≥ 60% inside the other, e.g. a face inside a YOLO head box.
- Every other YOLO box is kept: furry, anime and fursuit faces InsightFace can't see.
- A low-confidence InsightFace box (RetinaFace half-recognising a fursuit) is dropped where a YOLO box covers it, and kept elsewhere.
- *Largest Face* and the gender choices then pick from the merged list.
- Detections are cached per file, so the look-match probe, the look match's crops and the crop tools don't detect the same image twice in one session.
- *Test face detection on one photo* names the face finder it used.

YOLO models have **no gender**. In Auto mode, a gender choice applies to InsightFace's faces. With the finder set to YOLO, or when the selected face came from the YOLO model, the log says the finder can't tell gender on that face and the largest face is used.

## Auto look match

Embeddings from different models can't be compared, so the look match is decided **once per dataset / scoring session** and never mixed:

- The **Look Filter** decides from the training folder.
- The **gallery likeness** decides from its 3 baselines.
- **LoRA Royale** and **Repair Studio** decide from the reference photo (Repair Studio without one: from the baseline render).
- The **Block Profiler** decides from its subject photos.

How it decides:

1. Fizgig probes up to **24 images**, spread evenly over the sorted list, with the Auto detector.
2. An image counts as **human** if InsightFace finds a face with score ≥ 0.6. It counts as **YOLO-only** if its only faces came from the YOLO model.
3. The result:
   - **ArcFace** when at least **50%** of the probed images are human (`face_auto_threshold`) **and** fewer than **25%** are YOLO-only.
   - **DINOv2** otherwise: a furry or anime set, too few human faces, or a **mixed** set where both kinds are substantial. ArcFace can't embed a furry face, so a mixed set gets DINOv2 for all.

The decision is logged (the server console) and shown under the Look Filter's status line, for example:

```
[faces] Auto look match: InsightFace found human faces in 0/11 images, YOLO-only faces in 7/11 -> furry/anime set: DINOv2-base (face crop) on InsightFace + YOLO (fdetailer_v11.onnx) crops
```

The embedder is also recorded in the score caches (`embedder`, `embedder_label`), the gallery's status line, LoRA Royale's best-epoch line and Repair Studio's likeness chip (its tooltip). The decision is cached per set of probed files (path, mtime, size) and settings, so the same folder always decides the same way. An explicit look match (ArcFace / DINOv2 / CLIP), or an explicit finder, skips the probe.

**Why 50% / 25% / 0.6.**

- On our furry test images (drawn art and fursuit-con photos), InsightFace returns *some* face on 6 of 19 (32%): human bystanders, and RetinaFace half-firing on suits. Only 3 of those score ≥ 0.6 (16%).
- On the 47-image furry probe (huskies + drawn + suits), 4/24 were human and 15/24 YOLO-only → DINOv2.
- Human portrait crops: 9/9 human, 0 YOLO-only → ArcFace.
- Mixed humans + huskies: 9/22 human, 10/22 YOLO-only → DINOv2.
- A human dataset only fails the 50% bar if most of its images have no detectable face, and then ArcFace would score most of them as *no face* anyway.

### Fdetailer v1.1 (recommended furry model)

[Fdetailer – Adetailer for Furries v1.1](https://civitai.com/models/1228695) is an ultralytics YOLO **segmentation** model with 6 classes:

- `face`
- `penis`
- `pussy`
- `anus`
- `sheath`
- `pawpads`

Fizgig uses only the **`face`** class and ignores the mask head.

**Getting it.** Civitai lists the model as NSFW, and downloading it needs a free Civitai login. Fizgig does **not** download it for you. Download `fdetailerAdetailerFor_v11.pt` from the model page in your browser. If you use ADetailer in Forge/A1111, it's probably already in `models\adetailer`.

**Licence.** The Civitai page allows use without credit, derivatives, a different licence for derivatives, and commercial use including selling. So sharing a converted ONNX looks permitted. The checkpoint's embedded metadata says `AGPL-3.0`, which is ultralytics' default for anything trained with it. Fizgig neither ships nor auto-downloads the file. You convert your own copy once.

**Converting it to ONNX (one time).** Fizgig only loads `.onnx` files. A `.pt` file is a Python pickle, and loading one can run code. Converting needs ultralytics, which Forge/A1111 already have through ADetailer:

```powershell
cd C:\path\to\Fizgig
& "C:\path\to\webui_forge\system\python\python.exe" `
    src\fizgig\scripts\convert_yolo_to_onnx.py `
    "C:\path\to\models\adetailer\fdetailerAdetailerFor_v11.pt" `
    --out user_data\models\face\fdetailer_v11.onnx
```

Use any Python that has ultralytics: Forge's `system\python\python.exe`, A1111's `venv\Scripts\python.exe`, or a venv where you ran `pip install ultralytics`.

The script turns off ultralytics' auto-install and update checks, exports at 640×640, checks the result with onnxruntime when it's available, and prints the class it will use. 640 matched 1024 on our test set and runs about twice as fast. Then either save it as **`user_data\models\face\fdetailer_v11.onnx` inside the Fizgig folder**, as the command above does, where it's used automatically whenever the YOLO model box is blank, or type its path into **YOLO model** and press **Use**. `user_data/` is git-ignored.

**Defaults:**

- confidence **0.35**: low-confidence misfires on our set all scored below 0.4, while real faces scored 0.45–0.9.
- crop padding: the existing **20%**. Fdetailer boxes fit the face tightly.
- the look match crops the face with **25%** margin on every side.

### Built-in fallback: deepghs anime head v2.0_s (MIT)

When `face_yolo_model` is blank and there's no `user_data/models/face/fdetailer_v11.onnx`, Fizgig uses [deepghs/anime_head_detection](https://huggingface.co/deepghs/anime_head_detection) `head_detect_v2.0_s/model.onnx` (MIT licence, a ready-made ONNX, pinned revision). It downloads once and is then loaded from the Hugging Face cache, offline. To pre-fetch it along with the DINOv2 weights:

```
python -m fizgig.scripts.fetch_models --family tools
```

This model finds **heads**, so its boxes include ears and hair. That's useful for the look match, but crops come out looser than Fdetailer's.

### Other YOLO models

Any ultralytics detect or segment model exported to ONNX works. Fizgig handles all of these:

- detect `(1, 4+nc, N)`
- segment `(1, 4+nc+32, N)` plus protos
- end-to-end `(1, K, 6)`

Fizgig reads the class names from the ONNX metadata. It uses the first class named `face` (then `head`). A one-class model is used as is. If none matches, you get an error. Convert `.pt` files with the same script.

### Test results (22 Commons images: 11 drawn, 8 fursuit, 3 human controls)

Visual check, at conf ≥ 0.3 (Fdetailer: face class only):

| Model | Drawn (11) | Fursuits (8) | Humans | CPU per image |
|---|---|---|---|---|
| InsightFace buffalo_l | 0 | 0 | good | — |
| **Fdetailer v1.1** (640) | 9 (misses a chibi wolf and a mantis) | **7** (misses only a back view) | weaker (bus 1/2, Zidane 1/2) | 166–176 ms (i9, Fizgig's onnxruntime path) |
| toadboi FacesV1 | 9 | 5 | — | ~145 ms |
| deepghs head v2.0_s | 10 | 5–6 | ok | ~80–100 ms |

Fizgig's own onnxruntime decoder gives exactly the detections ultralytics gives with the same ONNX:

- Fdetailer on a Windows i9 PC: 26/26 face boxes, score difference 0.000.
- FacesV1, Bingsu face and person-seg on the test box: same result.

## Look match (identity embedders)

| Embedder | What it is | Match | Borderline | Floor |
|---|---|---|---|---|
| ArcFace (`buffalo_l` recognition) | human face recognition | 0.45 | 0.30 | 0.25 |
| **DINOv2-base** (`facebook/dinov2-base`, Apache-2.0) | general image features on the finder's face crop | 0.62 | 0.48 | 0.40 |
| CLIP ViT-B/16 (`openai/clip-vit-base-patch16`) | CLIP image embedding on the crop | 0.86 | 0.80 | 0.75 |

- *Match* and *borderline* set the green / orange / red verdicts in the Look Filter and the colour bands in the gallery.
- *Floor* is the lowest the Look Filter's Auto-Suggest cutoff can go. The cutoff is median − 1.5 × IQR, but never below the floor.
- Scores are only comparable **within one embedder**. A DINOv2 score of 0.6 and an ArcFace score of 0.6 mean different things.

**Calibration.** We calibrated DINOv2 and CLIP on these images:

- 29 face crops: one fursuiter across 11 photos, two drawn characters in two images each, a mascot pair, a fursona drawn in two styles, plus 15 different characters
- 14–25 same-character pairs and ~390 different-character pairs, with crops from FacesV1 and from deepghs head

| | Same character p10 / p25 / median | Different characters p75 / p90 / p95 |
|---|---|---|
| DINOv2 | 0.53–0.58 / 0.59–0.64 / 0.69–0.73 | 0.38 / 0.49–0.51 / 0.57–0.60 |
| CLIP | 0.80–0.86 / 0.81–0.89 / 0.88–0.92 | 0.71–0.75 / 0.76–0.80 / 0.78–0.83 |

We checked again with **Fdetailer** crops: 31 faces, 32 same-character pairs.

| | Same character p10 / p25 / median | Different characters p75 / p90 / p95 |
|---|---|---|
| DINOv2 | 0.61 / 0.69 / 0.75 | 0.39 / 0.53 / 0.60 |
| CLIP | 0.87 / 0.89 / 0.91 | 0.74 / 0.78 / 0.81 |

DINOv2 separates characters better, which is why it's the default for YOLO. CLIP is looser: different characters often score above 0.8.

These thresholds come from a small set. Treat them as a starting point, and trust the Look Filter's per-dataset fence (median − 1.5 × IQR) over the absolute numbers. DINOv2 also responds to hair, markings, ears and anything else inside the crop margin, which is usually what you want for a character.

**Speed.** Embedding takes roughly 0.2–0.4 s per image on CPU, including detection.

**Downloads.**

- ArcFace comes with InsightFace.
- DINOv2 (~350 MB of safetensors) downloads on first use or through `fetch_models --family tools`.
- CLIP downloads on first use.

### Cached scores record their embedder

- `fizgig_look_scores.json` (the Look Filter, read by the trainer's look-outlier warm-up) now stores:
  - `embedder` (e.g. `dinov2@auto:fdetailer_v11.onnx`, or `arcface`)
  - `embedder_label`
  - a `cutoff` computed on that embedder's scale (floored at its floor)

  The warm-up uses the cutoff as is and logs the embedder. Older files without these fields are ArcFace files and still work.
- The gallery's `likeness.json` stores `embedder`, `embedder_label` and `thresholds`, and the gallery draws its badge colours and chart bands from `thresholds`. When you reopen or resume scoring, saved scores are reused only if the embedder matches. Files without an `embedder` count as ArcFace. If you change the face settings mid-run, the gallery rescores from scratch.
- The in-memory embedding cache shared by the Look Filter, gallery and Repair Studio is keyed by the embedder's settings. Changing the finder, the model or the look match never mixes old and new embeddings.
- Block Profiler reports name the embedder in the likeness/bleed note.

## Limitations

- The clip-still picker (MiniMax H3 clips, RefMods) uses the same face finder, so with Auto it runs both detectors per candidate frame.

- YOLO faces have no gender. Gender targeting needs InsightFace, which only works on human faces.
- The Auto look match decides on a sample of up to 24 images (or on 1–3 reference photos for likeness sessions). A set that is, say, 40% human portraits and 60% faceless shots gets DINOv2. Set ArcFace explicitly to override, or lower `face_auto_threshold`.
- Auto detection costs both detectors on every image: about 0.3 s per image on CPU (~0.13 s InsightFace + ~0.17 s Fdetailer on an i9). The probe adds up to 24 images at the start of a scan; those detections are reused.
- Models with a dynamic input size, like the deepghs head ONNX, run at a fixed 640×640 letterbox. Ultralytics would use a rectangular letterbox, so scores can differ slightly (by up to ~0.2 on borderline boxes in our tests). Detections are the same in practice. Fixed-size exports, such as anything made with `convert_yolo_to_onnx.py`, match ultralytics exactly.
- Fizgig's NMS IoU is 0.5. ADetailer uses 0.7, so Fizgig drops more nested duplicate boxes. That suits *largest face*.
- DINOv2 and CLIP thresholds are provisional (small calibration set; see above).
- The identity checks run on CPU only. A large dataset takes a few minutes the first time; after that, embeddings are cached for the session.
