# Fizgig v5.2.0 — DRAFT (unreleased; grows until the next release)

## Context LoRA comes to MiniMax H3

Train a LoRA *on top of* an existing one. Pick any H3 LoRA in the Context LoRA box on the
Training tab — AI-Toolkit files load as-is — and it rides frozen under the one you're
training in every training step, then switches off for previews so they show the LoRA as
it deploys. Two ways to use it: a **training adapter** — Ostris's H3 assistant LoRA loads
as-is, on for training, off for sampling, exactly as AI-Toolkit runs it — or a style or
character LoRA the new one should learn to coexist with and be paired with at inference.
The saved LoRA records which file it was trained against and at what strength.

Verified on a real run: the same seed and prompt, before any training, renders the base
model's generic hooded skull without the context and the context LoRA's own character with
it — the context is genuinely part of what the new LoRA learns against. LoRA runs only;
fine-tuning refuses it up front.
