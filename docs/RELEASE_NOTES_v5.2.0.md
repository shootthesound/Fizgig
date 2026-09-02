# Fizgig v5.2.0

Faster, higher likeness on MiniMax H3 — from combining two methods that each fall short on their own.

## MiniMax H3: likeness blocks + the training adapter — the combination wins

Two ways exist to train a LoRA on H3 well, and until now they lived in different trainers. Fizgig's is **Optimised Likeness Learning**: H3's identity lives in the back 30 of its 50 blocks, so photo steps train blocks 20–49 only and leave the front of the model — composition, anatomy, prompt following — untouched. AI-Toolkit's, by **[@ostris](https://github.com/ostris)**, is the **[training adapter](https://huggingface.co/ostris/minimax_h3_training_adapter)**: H3 is guidance-distilled, so every plain-flow gradient is partly "learn the concept" and partly "undo the distillation"; a frozen assistant LoRA under the trainable one pulls the base back toward plain flow, so the gradient is all concept from the first step, and it's switched off for sampling.

We ran all three on the same dataset and seed — Fizgig's method alone, AI-Toolkit's alone, and both together — scoring every epoch's preview against the training photos with face recognition, 45 epochs each:

- **Each method alone landed in the same place** — a likeness-blocks run and an adapter run ended within a point of each other, taking about the same number of epochs to get there.
- **Together they got there a quarter sooner**, ran clearly ahead through the whole middle of the run, and finished higher than either.
- **Training all 50 blocks with the adapter — the adapter's strength with the block restriction off — was the weakest of the three late in the run.** Nothing wrong with the adapter: its cleaner gradient reaches every block it's allowed to, and for a likeness LoRA the front blocks have nothing to learn and a lot to damage. The block restriction is what keeps that cleaner gradient where it pays.

**In short: the combination reaches greater likeness and quality than either method does on its own.**

So the combination is now the default. The training adapter is a tickbox on the Training tab, **on in all three H3 presets** alongside Optimised Likeness Learning; it's active for every training step, off for previews, and never in your saved LoRA, so what you see and what you ship is the plain LoRA. The updater and the Preferences download button fetch both variants (fl2va and ref2va, ~155 MB each); the tickbox picks the one matching your Training Base. Its best window arrives earlier than you're used to — the likeness gallery is the place to pick the epoch. LoRA runs only; it steps aside under fine-tune.

@ostris — the block finding may be useful on your side too: the identity set (20–49) came out of Repair Studio ablations on real H3 LoRAs, and it holds for voice as well (the audio zone is 34–49). Thank you for publishing the adapters.

## Context LoRA comes to MiniMax H3

Train a LoRA *on top of* an existing one. Pick any H3 LoRA in the Context LoRA box on the Training tab — AI-Toolkit files load as-is — and it rides frozen under the one you're training, in every training step and in every preview, so the new LoRA learns to coexist with it and the previews show the pair the way you'll deploy it: a likeness on top of a style, an outfit on top of a character, a compatibility patch between two that fight. The saved LoRA records which file it was trained against and at what strength. Stacks with the training adapter above. LoRA runs only.

## Video clips follow likeness mode in LoRA runs too

The **Restrict video to likeness blocks** tickbox under Optimised Likeness Learning used to appear only for fine-tunes. It now applies to LoRA runs the same way — clip steps train the identity blocks (20–49) and leave the front of the model alone, exactly as photo steps do — and it's on by default with likeness mode in every H3 preset. Untick it for whole-model video. Verified on a clip-only run: blocks 0–19 never moved.

## Small things

- "Medium to High LR" is now **Medium to High Noise LR** — it scales the learning rate of the noisy-half steps.
- The MiniMax training hints are one line each, with a pointer to the README's MiniMax section.
