# Fizgig v5.2.0 — DRAFT (unreleased; grows until the next release)

## MiniMax H3: the training adapter — faster, higher likeness in one tickbox

H3 is a guidance-distilled model, and training a plain LoRA against it means every
gradient is partly "learn the concept" and partly "undo the distillation". Ostris's
[training adapter](https://huggingface.co/ostris/minimax_h3_training_adapter) — the
assistant LoRA AI-Toolkit trains with — pulls the base back toward plain flow so the
gradient is all concept from the first step. Fizgig now ships it as one tickbox on the
Training tab: on for every training step, off for previews, never in your saved LoRA.

Measured, same dataset and seed, likeness mode on: **50% likeness seven epochs sooner
(23 vs 30), a higher peak (61% vs 57%), and +13 points across epochs 10–20.** Its best
window arrives earlier too — the likeness gallery is the place to pick the epoch. All-blocks
training with the adapter was worse than either, so keep Optimised Likeness Learning on.

The updater and the Preferences download button fetch both variants (fl2va and ref2va,
~155 MB each); the tickbox picks the one matching your Training Base. Thank you
**[@ostris](https://github.com/ostris)** for publishing the adapters. LoRA runs only.

## Context LoRA comes to MiniMax H3

Train a LoRA *on top of* an existing one. Pick any H3 LoRA in the Context LoRA box on the
Training tab — AI-Toolkit files load as-is — and it rides frozen under the one you're
training, in every training step and in every preview, so the new LoRA learns to coexist
with it and the previews show the pair the way you'll deploy it: a likeness on top of a
style, an outfit on top of a character, a compatibility patch between two that fight. The
saved LoRA records which file it was trained against and at what strength. Stacks with
the training adapter above.

Verified on a real run: the same seed and prompt, before any training, renders the base
model's generic hooded skull without the context and the context LoRA's own character with
it — the context is genuinely part of what the new LoRA learns against. LoRA runs only;
fine-tuning refuses it up front.
