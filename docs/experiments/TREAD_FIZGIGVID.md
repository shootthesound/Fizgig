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
