# Fizgig v5.2.0 — DRAFT (unreleased; grows until the next release)

## Context LoRA comes to MiniMax H3

Train a LoRA *on top of* an existing one. Pick any H3 LoRA in the Context LoRA box on the
Training tab — AI-Toolkit files load as-is — and it rides frozen under the one you're
training, in every training step and in every preview, so the new LoRA learns to coexist
with it: a likeness on top of a style, an outfit on top of a character, a compatibility
patch between two that fight. The saved LoRA records which file it was trained against and
at what strength; pair them the same way at inference.

Verified on a real run: the same seed and prompt, before any training, renders the base
model's generic hooded skull without the context and the context LoRA's own character with
it — the context is genuinely part of what the new LoRA learns against. LoRA runs only;
fine-tuning refuses it up front.
