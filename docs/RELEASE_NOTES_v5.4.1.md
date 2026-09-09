# Fizgig v5.4.1

Weight averaging is on by default for MiniMax H3, and it's measurably better.

## EMA 0.98 is now the recommended setting, and the default in every H3 preset

Weight averaging (EMA) saves each checkpoint and preview from a running average of the adapter's recent steps rather than from whichever step the epoch happened to end on. At batch size 1 the raw weights lean toward the last few images they saw, so a checkpoint carries the tail of the shuffle; the average carries the whole dataset. It was there before as an option for runs pushed hard. It's now on for everyone.

The numbers, from a four-way A/B on the same dataset and seed, 50 epochs each, scored by the gallery's likeness meter:

| Setting | Likeness, epochs 41–50 | Spread between epochs | Epochs at 60%+ |
|---|---|---|---|
| Off | 56.5% | 3.9 | 6 |
| **0.98** | **61.5%** | **1.6** | **14** |
| 0.99 | 58.2% | 3.6 | 7 |
| 0.995 | 54.5% | 4.3 | 3 |

0.98 sits five points higher on the late epochs with less than half the wobble between them, and reaches 50% likeness just as early as EMA-off. 0.99 smooths without lifting the level. 0.995 averages over so many steps that it lags the run and finishes below off. So the dropdown now reads **0.98 (recommended)**, **0.99 (stronger)** and **0.995 (long runs only)**, every H3 preset ships 0.98, and Off is there for your own A/B. A run resumed after a pause keeps its average. If you had 0.99 saved, it stays 0.99.

## Previews on tight RAM: a warning instead of a crash

On a card below 32 GB each preview parks the training base into system RAM and streams the text encoder through it, which on a 32 GB-RAM machine can exhaust Windows' memory commit — the app then closes with a "not enough memory resources" dialog, or an RDP session drops, with nothing in the training log to explain it. The trainer now says so at launch when previews are on, the card is under 32 GB and RAM is under 40 GB: enlarge the paging file, lighten the samples (a smaller canvas, 22 frames rather than 56, or a still), and as a last resort run without sample generation and judge checkpoints in LoRA Royale. The README's requirements line says the same.
