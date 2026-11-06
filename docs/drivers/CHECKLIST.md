# Before you ship a family

A family is ready when it trains, previews and saves correctly, every ability the model can support is switched on and does what the GUI says, its files download, and the docs name it. "It runs" isn't the bar: most driver bugs produce a run that finishes and a LoRA that quietly does less than it should, and a family that only does what its pull request set out to do is missing what people expect on every model. These checks catch both.

**The rule behind every check: a new model works like the models already in Fizgig.** Users move between models and expect the same Training tab, the same presets, the same controls and the same files. So a family gets every shared ability its model can support, and nothing the other models don't have: no controls, presets, notes, hints or settings of its own. The exception is when the model offers something the others don't, and the only way to give it to people is a control of its own. Then add that control, keep it in plain words like the rest of the tab, and put a comment beside it saying what it gives. Changes people never see, such as saving LoRAs in the layout ComfyUI needs, are fine wherever the model needs them, also with a comment saying why. If you're unsure whether something belongs, look at what Qwen Image 2.1 and Flux 3 do and do the same.

- [ ] A Discussion for the model is open in the repo (see [README.md](README.md#before-you-start-open-a-discussion)), and the scope agreed there.
- [ ] **The pull request stays in its scope:** `git diff --stat master` lists your description, driver and model package, and `registry.py`. Any other file is a default-off hook your model needs, named in the pull request's description with what it leaves unchanged for the other families. A fix to shared code goes in a pull request of its own ([README.md](README.md#what-a-driver-doesnt-bring)).
- [ ] `python -m pytest checks -q` passes.

## The description

- [ ] `description.validate()` returns `[]` (the registry refuses to import a family that fails it).
- [ ] Every value from outside Fizgig has a `source=` or a comment naming it.
- [ ] `driver.block_map()` works with no model loaded (the Profiler and Extract call it that way).
- [ ] **It runs offline once its files are on disk:** `python src/fizgig/families/offline_check.py --family <key>` passes. It confirms every `helper_files` repo is cached, then loads the text encoder (encoding a prompt) and the VAE with the Hub switched off. A tokenizer or config loaded with plain `from_pretrained` fails here; use the driver's `from_pretrained` / `helper_dir` instead.
- [ ] A family sampled with CFG sets `workbench_follows_samples=True`, so the workbench encodes the Samples tab's negative; without it the driver gets no negative and must supply its own.
- [ ] **No licence text in the app.** No family shows its model's licence in the GUI: not in `model_note`, a hint, a preset note or Preferences. The licence goes in the family's guide (`docs/<FAMILY>.md`) and nowhere else. `model_note` is for how to use the model, and most families leave it empty.
- [ ] **`preview_negative` is the reference pipeline's own default**, read from its code, not assumed empty: Fibo 2.0's pipeline uses an empty structured JSON caption when no negative is given.

## Downloads

- [ ] **Download models for me works without a token:** `python -m fizgig.scripts.fetch_models --family <key> --dry-run` lists every default file, and each one downloads without a Hugging Face token whenever an ungated copy of the same file exists. A gated upstream repo is no reason to make users sign in: Comfy-Org's repo for the model usually carries the same build ungated, and a text encoder shared with another family is often already in that family's repo.
- [ ] **What the model's own repos ship is used:** list the files of the model's ComfyUI repo and its upstream repo (`HfApi().model_info(repo, files_metadata=True).siblings`). Speed / Turbo LoRAs go in `speed_loras`; int8 or fp8 builds of the DiT the loader can take go beside the default as `alt_repo` / `alt_path`; a distilled or Turbo checkpoint is a preview checkpoint; an edit checkpoint is a sign the model supports edit training. Each one you leave out gets a comment saying why.
- [ ] Whatever the default run uses (a speed LoRA on by default, a training adapter) is a default download, not an optional one.

## Training

- [ ] **Parity with the reference implementation:** from the same starting noise and prompt, your `generate` matches the model's reference pipeline (diffusers, ComfyUI or the original repo) to bf16 precision. Measure it; a picture that "looks right" hides schedule and conditioning bugs. Compare stage by stage (text conditioning, one model prediction, the sigma schedule) from identical inputs. If your model code isn't the reference's code, compare in fp32: two correct bf16 implementations can differ by 1–2% per step, which compounds into a visibly different render.
- [ ] A short run through the command line (cache latents, cache text, train two epochs with previews) finishes, and the previews move toward the training data.
- [ ] **The clean-end share is measured:** the share of sampled training timesteps below 0.5, from 100,000 draws, is in the pull request. Below about 20%, check a likeness run for slow likeness and previews whose composition jumps between epochs ([STILLS.md](STILLS.md#the-training-step)).
- [ ] **The LoRA loads in ComfyUI with every module.** ComfyUI's console lists any key it can't map ("lora key not loaded"); there must be none. The LoRA then changes ComfyUI's output the way the previews show.
- [ ] **Every file Fizgig writes loads in ComfyUI**, not only the training save: Studio's Save Repaired, Extract's output and a fine-tune checkpoint too. Where ComfyUI's checkpoint fuses Linears the model trains split (Fibo 2.0's `qkv` / `to_kv` / `w13`), set `LoRAFormat.save_fused` and give the driver `export_lora_state_dict` and `convert_lora_state_dict`; every writer then goes through them. Check that the converter also splits another trainer's fused LoRA exactly, not only your own block-diagonal files.
- [ ] Every built-in preset trains one epoch through the GUI (Load Preset, Start Training) and saves a LoRA.
- [ ] `train_memory` figures are measured, at two resolutions per precision, from the trainer's logged peaks, previews included, **with the family's default preview setup** (its speed LoRA on, if that's the default) and with it off; each figure is the higher of the two. The same goes for `compile_memory` and, on a 40 GB+ card, `train_memory_nogc` and `train_memory_partial`.
- [ ] **Override next sample**: tick it mid-run with a new prompt, seed and size; the next preview renders that, and unticking it brings back the Samples tab's prompts. A slider run ignores it.
- [ ] Pause and Resume: a run paused mid-way and resumed continues from the same epoch, and its loss carries on from where it was.

## Small cards

Most people train on 12–24 GB cards, so a family isn't finished until it has been run on them. Nothing here is optional when the family offers the feature; `FIZGIG_SIM_VRAM_GB=N` makes the planner and the allocator behave as an N GB card, so a large card can run every tier.

- [ ] `train_memory` is measured (above), so Auto has real figures to plan from.
- [ ] **Auto plans fit:** a LoRA run with Auto precision and Auto block swap finishes, previews included, under `FIZGIG_SIM_VRAM_GB=12` and `=16`. Note the plan it logs (`[precision] Auto plan: ...`) and the peak.
- [ ] **Block swap runs**, if the family has it: one run with swap forced on (e.g. bf16 with 10 blocks swapped under a simulated 16 GB card) trains, previews and saves. Built is not the same as working.
- [ ] **Fine-tune plans fit**, if the family fine-tunes: `ft_spec`'s memory figures (`overhead_gb`, `stream_base_gb`, `calib_mp`, `act_gb_per_mp`) come from measured window peaks at two resolutions, and a fine-tune finishes under `FIZGIG_SIM_VRAM_GB=16` and `=24` with the fewest windows that fit.
- [ ] **The workbench renders on a small card:** a Repair Studio render with Auto precision and swap under `FIZGIG_SIM_VRAM_GB=12`.
- [ ] The text encoder fits the smallest card you support on its own (it runs alone while caching); if it doesn't, give it a quantised fallback as Qwen 2.1's does.
- [ ] **Its other builds are indexed:** every published build of the text encoder that the driver's loader takes is in `text_encoder_builds`, measured with `scripts/check_te_builds.py` ([Text encoder builds](README.md#text-encoder-builds)), so helpers can fetch the exact build a run used and a lighter one to try prompts with.

## Workbench

- [ ] Each tab named in `workbench` loads the family and renders: Repair Studio (I'm feeling lucky included), the Profiler (a quick profile writes its report), Extract (a rank-reduced file), LoRA Royale (two checkpoints, seed travel, and prompt travel - it needs `pad_conditioning`, and without it Royale quietly offers no prompt travel).
- [ ] **Save Repaired is faithful:** a LoRA saved with some blocks switched off renders the same as the live preview it was saved from.
- [ ] With `activation_cache`: a slider change renders the **same picture** with Reuse unchanged blocks on as with it off.

## Abilities: every one the model can support

Switch on every ability the model can support, and test each one below. Leave one out only when the model can't do it or a measurement shows it hurts, and name it in the description's `not_offered` with the reason. [ABILITIES.md](ABILITIES.md) lists what each one asks of the driver.

`checks/test_family_abilities.py` enforces this: a picture family that lacks an ability at least half the picture families have, without a `not_offered` entry, fails the checks. Run them before calling a family done.

- [ ] **LoKR** (`network_types`), or a comment saying why not.
- [ ] **Masked training** (`masked_training`): a run with masks and the prior outside them trains and previews.
- [ ] **Slider LoRAs** (`slider_training`), from photo pairs and from prompts, with a rank-8 Slider preset.
- [ ] **Edit LoRAs** (`edit_training`) wherever the model takes reference images, with an Edit preset.
- [ ] **Fast previews:** a speed LoRA, distilled model or preview checkpoint wherever one is published.
- [ ] **Reuse unchanged blocks** (`activation_cache`).
- [ ] **Prompt travel** (`pad_conditioning`) and seed travel in LoRA Royale.
- [ ] **Compile** and **INT8 attention** where they measure faster.
- [ ] **EMA**, **Adaptive LR**, the **per-image loss watch**, **caption dropout** and **Multi Concept**, unless the model has a reason against.
- [ ] **Batch size** (`batch_training`): `training_loss_batch` (and `stack_conds` when captions vary in length), checked on the GPU against the one-picture loss from the same generator state, with captions of different lengths in the batch.
- [ ] **`start_lr_bands`** for every kind the family offers.
- [ ] **No Model Area to Train** (`train_areas` stays `()`) and no Fast Identity Mode (`identity_blocks` stays `()`), as on every model but Klein (Model Areas) and Qwen Image (Fast Identity Mode): the LoRA trains the whole model. Areas and identity blocks come only after the block map has been charted by measurement (what each block group does), as Klein's and Qwen Image's were, never from a guess or from what a test run happened to train, and they are added in their own change, not with the family.
- [ ] **Presets are the standard set**, named after the model: a LoRA preset on the whole model, plus Edit, Fine-tune and Slider where the family has them. No preset that leaves blocks out.

The abilities that need more than a run to check:

- [ ] Fine-tune (optional; a family can ship without it): a run at learning rate 1e-30 saves a file equal to the source, tensor for tensor (only weights that were exactly zero may move). Then one real run, loaded in ComfyUI in place of the base, and a small-card run under `FIZGIG_SIM_VRAM_GB`. See [FINETUNE.md](FINETUNE.md).
- [ ] Edit training: an edit LoRA trained on a few pairs applies the edit to a new photo.
- [ ] Sliders: at strength −1, 0 and +1 the previews move between the two looks, and 0 matches the base model.
- [ ] Video: an off-spec clip (wrong frame rate or frame count) is refused at Start with the reason; a clip preview plays with sound (if the model has sound); Repair Studio plays the tweaked and baseline clips.
- [ ] Start checks: `start_lr_bands` declared for each kind the family offers, from its presets (a third of the lowest to three times the highest); a preset's own rate gives no warning at Start.

## Docs

- [ ] A guide, `docs/<FAMILY>.md`: setup, presets, memory, the abilities, ComfyUI, licence, the command line.
- [ ] `docs/CLI.md`: the contents, the model-files table, the `--family` list and a training section with a full example.
- [ ] `README.md`: the title, the intro, the models badge, the Supported models table, and the slider / edit / fast-preview lists it belongs in.
- [ ] `docs/INSTALL.md` (the GPU line, a download section and a VRAM section), and `docs/FINETUNE.md` / `docs/TRAINING.md` where they list models.
- [ ] `docs/THIRD_PARTY_NOTICES.md` for any code you ported.

## Before you call it done

- [ ] **Compare the description with the other families' in both directions.** First, every field Qwen Image 2.1 or Flux 3 sets that yours leaves at its default either gets switched on or gets a comment saying why not:

  ```python
  import dataclasses
  from fizgig.families import registry
  mine, ref = registry.get("<key>"), registry.get("qwen_image21")
  for f in dataclasses.fields(mine):
      d = f.default if f.default is not dataclasses.MISSING else (
          f.default_factory() if f.default_factory is not dataclasses.MISSING else None)
      if getattr(mine, f.name) == d and getattr(ref, f.name) != d:
          print(f.name)
  ```

  Then the reverse: every field yours sets that at most two other families set is something only your model has. Each one is removed, unless it gives people something only this model offers, or it is a change they never see that the model needs; either way, its comment says so. Check `ModelFile` fields and the driver's methods the same way:

  ```python
  fams = {k: registry.get(k) for k in registry.FAMILIES}
  for f in dataclasses.fields(mine):
      d = f.default if f.default is not dataclasses.MISSING else (
          f.default_factory() if f.default_factory is not dataclasses.MISSING else None)
      others = [k for k, o in fams.items() if o is not mine and getattr(o, f.name) != d]
      if getattr(mine, f.name) != d and len(others) <= 2:
          print(f.name, others)
  ```

## What a finished family includes

The checks above prove the family works. These make it good, and every family should have them unless it has a measured reason not to. Most are one field in the description.

**Every ability your model can support.** People expect the same tools on every model, so switch on each ability the model is able to do, not only the one you came for. Most cost a field or two: slider LoRAs (`slider_training`: photo pairs need nothing more, prompt sliders two short methods), LoKR in `network_types`, EMA (`ema_default`), the per-image loss watch, a speed LoRA for fast previews, Reuse unchanged blocks in Repair Studio (`activation_cache`), every workbench tab, compile and INT8 attention where they measure faster, edit training if the model takes reference images, and full fine-tuning (which can also come later). Leave one out only when the model can't support it or a measurement shows it hurts, and say why in a comment beside the description. [ABILITIES.md](ABILITIES.md) lists them all.

**Research before presets.** Before writing presets or sampling defaults, find what the model's users actually settled on, not only the model card: the sampler, scheduler, steps and CFG; the learning rate, rank and alpha, epochs or steps per dataset size, and optimizer; and which layers must never be trained (Anima's LLM adapter, for example). Cite the sources in comments beside the values.

**Parity with the reference.** Measure your text conditioning, one model prediction and the sigma schedule against the reference implementation, from identical inputs. If your model code isn't the reference's own code, compare in fp32. A config file written for a newer library can be misread silently; spell such values out in the driver.

**Presets.**
- Use `muon` for LoRA presets, with `"OPTIMIZER_ARGS": MUON_DEFAULT_ARGS` (Peter, 6 Nov 2026: every picture model's presets are on it, for its source-matching detail). Muon's step is matched to AdamW's (`adjust_lr_fn=match_rms_adamw`), so pick learning rates as for AdamW (one Qwen Image 2.1 run overcooked when Adaptive LR took Muon to 3.5e-4). Add the new family's key to `CHOSEN` in checks/test_optimizers.py, or that check fails. `adamw8bit` and `adamw` stay in the optimizer list: if a model's LoRAs melt on Muon, use one of them for that preset and say why in the description (on 8-bit, set `optimizer_eps_floor_8bit` if some tensors step far past the learning rate, as MiniMax H3's did). Fine-tunes ignore the preset's optimizer and use Adafactor.
- Set `ema_default="0.98"`, as every other family does.
- Use a flat learning rate where the per-epoch loss is noisy, and set `adaptive_lr=False` if the plateau detector would only react to noise (SDXL).
- Set epoch counts from the community's step targets for a typical dataset, and save every epoch.
- Ship only presets you've trained.

**Speed.** Time a real run through the GUI, not an isolated loop. If the GPU sits idle, the step is launch-bound: set `compiles=True` and measure `compile_payback_steps`. If your blocks don't sit in one `ModuleList`, or are called with keyword arguments, override `compile_blocks` and call `families/compile.ready_to_compile()` first (SDXL, Anima). Try `int8_attention=True` with the model's attention routed through `attend()`, and keep it where it measures faster (Anima 1.13x); a launch-bound model gains nothing (SDXL).

**Memory.** Fill `train_memory` from measured peaks at two resolutions per precision, previews included: the preview often sets the peak.

**Samples tab.**
- Give a family that previews above CFG 1 a default negative (`preview_negative`). Leave it `None` for a CFG-free family, so the box greys out.
- If the model is sensitive to its sampler, name the sampler and schedule beside the Steps box, with the ComfyUI names (`samples_text` "sampler").
- If a speed LoRA defaults to off (`preview_speed_strength=0`), set `preview_speed_steps` to the plain model's step count, not the speed LoRA's.

**Workbench.** For a 1 MP model, start Repair Studio at 1024 (`repair_size=1024`). Repair Studio's own negative box appears by itself for a family with a default negative that renders above CFG 1.

**Other people's LoRAs.** Read the keys of a few public LoRAs for the model (Hugging Face serves the safetensors header alone) and check every layer resolves. Map other trainers' names in `alias_flat`. Conv layers in LoCon and UNet speed LoRAs load on their own.

**Long prompts.** If the text encoder has a token limit, handle long prompts the way ComfyUI does (SDXL encodes 77-token chunks side by side) rather than cutting them off.

**Sliders.** Slider LoRAs take about 40 lines in the driver: `noise_latents` and `predict`, and the pair weighting in `training_loss`. Then set `slider_training=True`.

## The pull request

Say which model and which files, the VRAM the training run needed, and link the sources behind the description's values. Please open a discussion before starting a family, so two people don't build the same one.
