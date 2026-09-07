# TREAD and TREAD Plus FizGigVid — timing (experiment/tread, 6 Sep 2026)

RTX 5090, MiniMax H3 int8 base, rank-8 LoRA, training adapter on, likeness mode default, previews off. Step time is the trainer's own it/s over 24 steps (3 epochs of the 8-clip `clipq2` set at 0.25 MP, 384×640 bucket); photo numbers are 58 steps of the 29-photo `mbacc` set. Quality is NOT measured here — that is the A/B, judged by eye and ArcFace, with plain training at a lower Target Megapixels as the honest baseline.

| Run | s / step | speed-up | avr_loss |
|---|---|---|---|
| Photos 0.25 MP, off | 0.61 | — | 0.396 |
| Photos, TREAD 0.5 @ 2–47 | 0.45 | 1.35× | 0.464 |
| 22-frame clips, off | 3.44 | — | 0.550 |
| 22-frame, FizGigVid 2× throughout | 1.48 | 2.3× | 0.583 |
| 22-frame, FizGigVid 4× front / 2× identity (default) | 1.32 | 2.6× | 0.593 |
| 22-frame, FizGigVid 4× throughout | 1.12 | 3.1× | 0.621 |
| 22-frame, default + TREAD 0.5 inside | 1.17 | 2.9× | 0.832 |
| 56-frame clips, off | 10.20 | — | 0.547 |
| 56-frame, default preset | 2.88 | 3.5× | 0.592 |

Notes: the running loss rises with the amount of thinning, as expected (thinned tokens are predicted from a coarser state) — it is not a quality signal. The gains are below the pure-compute estimates (≈5× for the default on 22 f) because a step also carries the optimizer, the checkpoint recompute of everything outside the blocks, and data loading; they grow with clip length, as attention's share does. FizGigVid does nothing on photo steps by design.

## Clip first frame as a photo — the slice is the still (7 Sep 2026)

The feature slices a clip's cached latent at frame 0 instead of encoding the frame again. Checked on the real H3 video VAE encoder (5090, fp32, 22 random frames at 128×160): the clip latent's first frame against the first frame encoded alone differs by max 3.4e-3, mean 1.9e-4, against a mean |z| of 0.85 — 0.02% on average, numerical noise from the causal stack. Frame 1 also matches between a 22-frame and a 5-frame encode to 1e-2, i.e. later frames do not reach earlier latents.

## Clip still = the sharpest frame with a face (7 Sep 2026)

Frame 0 is replaced by a picked frame. At cache time (`minimax_cache_latents --clip_still`, which the GUI passes when the tick is on) every clip's frames are scored by the variance of the Laplacian; candidates are visited sharpest-first and run through the app's InsightFace detector (CPU, with the Look Filter's pad-and-retry for frame-filling faces) until six frames with a face of at least 8% frame height are found; the winner is the one with the sharpest FACE CROP. That frame is encoded as an ordinary still into the clip's own cache file (`still_latent`, `still_frame`); the training item loads it directly. No face anywhere → frame 0, flagged in the log. Measured on the 23 videotest clips: ~0.3 s per clip on CPU (6 detections), 25 s for the whole cache pass on the 5090 including the clip re-encode; every clip found a face (two needed the pad retry). With `--skip_existing` only clips lacking a pick are re-encoded, so a second pass encodes nothing. A clip cached before the tick was on falls back to frame 0 with a warning at dataset build.
