# Fizgig v5.7.0

MiniMax H3 gets a second recipe. Fast is what you have been running. Ultra quality gives better likeness and better audio, at a slower step.

## Training mode: Fast, Ultra quality, or Off

The Optimised Likeness Learning tickbox on the Training tab is now a **Training mode** dropdown, and it works within every preset: load any preset, then choose the Fast or the Ultra quality version of it.

- **Fast** is the recipe you know. The quickest steps, good on picture and sound. The character presets load with it selected.
- **Ultra quality** trains the whole of the model that matters. On the same dataset and seed it produced the best likeness this trainer has managed, and to the eye and ear it is clearly ahead of Fast: sharper faces, cleaner audio, and the dataset's own quirks stay out of the LoRA for far longer. Steps are slower. It is also the mode for style and scene work, and the Style preset now loads with it selected.
- **Off** is for hand-picking blocks yourself, for experiments.

Saved presets and Load Settings From Last Train carry over: a run saved with the tickbox on restores as Fast, one saved with it off restores as Off with its blocks intact.

## A more accurate memory plan

The automatic VRAM plan now counts everything that sits on the card for the whole run. Most runs will not notice. Where a run was being planned tighter than the card could hold, it now gets a plan that fits, and the console says what to change to get the more accurate base back.

## AMD ROCm on Windows

Two contributions from **[@scryptio](https://github.com/scryptio)**, thank you. The launcher now detects the card at start: RX 5000 and 6000 series cards run the standard attention path instead of the experimental kernels, which is the stability-first choice for those GPUs, and newer cards are unchanged. It also points rocBLAS and MIOpen at their library databases, which quiets the rocBLAS warnings some users saw at start-up. And the installer's `--experimental` option now draws from AMD's newer whl-next nightlies; the pinned default install is untouched.

## Fixed

- Load Settings From Last Train on the MiniMax tab could show a Krea 2 tickbox that does not belong there. Gone, along with a matching stray row on the Klein tab.
