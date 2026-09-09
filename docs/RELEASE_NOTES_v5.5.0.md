# Fizgig v5.5.0

Weight averaging comes to Krea 2, on by default, and it measures the same way it did on H3.

## Weight averaging (EMA) on Krea 2 — on by default at 0.98

Checkpoints and previews are saved from a running average of the adapter's recent steps rather than from whichever step the epoch happened to end on, so every checkpoint reflects the whole dataset instead of the tail of the shuffle. It shipped on MiniMax H3 in 5.4.1; it is now on Krea 2 too, with the same result on a real A/B: a steadier climb through the epochs and a higher late-epoch likeness than the raw weights give.

It's a new row on the Training tab, in Training Parameters, shown under Krea 2: **Weight averaging (EMA)**, reading **0.98 (recommended)**, with 0.99 (stronger), 0.995 (long runs only) and Off for your own A/B. Every Krea 2 preset ships 0.98. Training itself runs on the raw weights; a paused run keeps its average and picks it up on resume; every checkpoint records the decay in its metadata. Under fine-tune rotation there is no adapter to average, so the setting is ignored with a note in the console. Klein is unchanged.

## LoKR on Krea 2: the learning rate is the answer

LoKR works well on Krea 2 and standard LoRA stays the default. The one thing LoKR needs is a lower learning rate: **5e-5** whichever preset you started from, or with Adaptive LR a **Min of 5e-5 and a Max of 1e-4**. The line under Network Type now says exactly that the moment you select LoKR, and goes back to the plain trade — LoKR potentially higher quality, LoRA about 20% faster — when you select LoRA.

## A halfway resolution: 0.37 megapixels

Target Megapixels gains **0.37** between 0.25 and 0.5, on the Training tab and in Image Prep. It's a 608-pixel side on every family's bucket grid, the midpoint in latent resolution between 0.25 (512) and 0.5 (704), for when 0.5's extra detail is worth having but not at twice the step time. The memory planners take it in their stride; nothing else changes.

## Fixed: a checkpoint's thumbnail was the previous epoch's preview

On Klein and Krea 2 the epoch checkpoint is written a moment before that epoch's preview renders, and the automatic thumbnail took whatever preview was newest on disk, so every checkpoint carried the neighbouring epoch's picture. The weights were always right; the picture was one epoch stale. Now, once the epoch's own preview has rendered, the checkpoint gets it embedded — chosen by epoch number, tensors untouched. A thumbnail you set yourself is never overridden. Reported by **[@DigitalBeer](https://github.com/DigitalBeer)** (#122), thank you.
