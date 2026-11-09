# Training maps: progress (local branch experiment/training-maps, never pushed until Peter has seen it)

Started 9 Nov 2026 ~03:45, overnight (no questions; decisions mine). Peter's ask: open the black box of training - see
on the picture where the model is still wrong and where the LoRA is acting, step by step.

## The design (agreed in chat)
Two maps that need no attention internals, so they work for every picture family:
1. **Where it's still wrong** (per-pixel loss): for a few fixed probe pictures at fixed noise levels and a fixed seed,
   each latent position's squared error (prediction vs the training target), channel mean.
2. **Where the LoRA changes it** (LoRA effect): the same input predicted with the LoRA at 0 and at 1, |difference|.
Trigger-token attention is deliberately left out for now: the Qwen-family text encoders are causal, so every token after
the trigger carries it and attention to the trigger's position would mislead. Maybe later, as an experiment.

## How it's built
- Every family already has `driver.noise_latents()` + `driver.predict()` (the textual sliders use them) and the network
  has `set_trainable_multiplier()`, so both maps run through existing hooks.
- Target: new driver hook `map_target(latents, state)`. Default (flow matching, x_t = (1-t)x0 + t*noise, target
  noise - x0) = (x_t - x0)/t. Overrides: MiniMax (x0 - noise), SDXL (eps or v).
- Probe timesteps 0.25 / 0.5 / 0.75 (avoid tiny t: the target divides by t).
- Fixed noise per probe + timestep (seeded generator); RNG state saved and put back; model in eval; block swap
  streamed as inference (`driver.block_swap_mode`); EMA swapped in when the run has EMA (what previews and saves use).
- A baseline before training (epoch 0: the base model's own error; the effect map is zero there), then each epoch.
- Output: `<output>/training_maps/<LoRA name>/maps.jsonl` (runs share output folders), one line per (epoch, probe, t, kind) with a small grid (pooled to <= 48
  on the long side); `probes.json` names the probe pictures. The page draws them over the picture with one colour
  scale per probe+t+kind across all epochs (a per-frame scale would hide the trend).
- A family ability `training_maps` (description), on for still families, off for video/audio for now.
- UI: a "Training maps" card on the Samples tab: probe, noise level, which map, an epoch scrubber and play.

## Status
- [x] driver hook map_target (default flow; SDXL override; MiniMax/Vimoo/ACE-Step have the ability off - clips/songs)
- [x] training/train_maps.py (probes, capture, write)
- [x] train.py hook (baseline + each epoch), argparse flags
- [x] description ability (+ checks/test_family_abilities) + form.py row TRAINING_MAPS + launch.py flag
- [x] server endpoint /api/runs/train_maps (meta, then one picture + level at a time) + viewer: a Training maps button beside Per-image loss on the run card and in Runs & outputs (not a Samples card - follows the per-image loss pattern). Views: Still wrong / Learned since the start (baseline - now, green/red) / Where the LoRA acts
- [x] local functional test (MOSS, 6 pictures, 0.25 MP, 3 epochs, 5090): ~2 s an epoch; effect map gathers on eyes/features after 2 epochs; raw loss map is mostly speckle (one noise draw), hence the Learned view
- [x] Krea 2 local test: its predict() returns patch tokens - fixed generally (train_maps.to_map lays tokens on the patch
  grid; Krea 2 / Flux 3 / Qwen give map_target in token layout, Anima with a frame axis, Fibo from its stored noise).
  Krea 2 maps line up (effect on the eyes, as MOSS)
- [ ] one-epoch sweep of the other still families on the 5090: klein, qwen_image21, zimage, flux3, anima, fibo2, sdxl
- [ ] the viewer in a browser (Peter: no headless captures - he checks it)
- [x] docs (WEB_UI.md)
- [ ] CLAUDE.md note (when merged)
