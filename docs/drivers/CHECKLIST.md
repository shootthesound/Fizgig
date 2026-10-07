# Before you ship a family

A family is ready when it trains, previews and saves correctly, and every ability it declares does what the GUI says. "It runs" isn't the bar: most driver bugs produce a run that finishes and a LoRA that quietly does less than it should. These checks catch those.

- [ ] A Discussion for the model is open in the repo (see [README.md](README.md#before-you-start-open-a-discussion)), and the scope agreed there.

## The description

- [ ] `description.validate()` returns `[]` (the registry refuses to import a family that fails it).
- [ ] Every value from outside Fizgig has a `source=` or a comment naming it.
- [ ] `driver.block_map()` works with no model loaded (the Profiler and Extract call it that way).
- [ ] A family sampled with CFG sets `workbench_follows_samples=True`, so the workbench encodes the Samples tab's negative; without it the driver gets no negative and must supply its own.

## Training

- [ ] **Parity with the reference implementation:** from the same starting noise and prompt, your `generate` matches the model's reference pipeline (diffusers, ComfyUI or the original repo) to bf16 precision. Measure it; a picture that "looks right" hides schedule and conditioning bugs. Compare stage by stage (text conditioning, one model prediction, the sigma schedule) from identical inputs. If your model code isn't the reference's code, compare in fp32: two correct bf16 implementations can differ by 1–2% per step, which compounds into a visibly different render.
- [ ] A short run through the command line (cache latents, cache text, train two epochs with previews) finishes, and the previews move toward the training data.
- [ ] **The LoRA loads in ComfyUI with every module.** ComfyUI's console lists any key it can't map ("lora key not loaded"); there must be none. The LoRA then changes ComfyUI's output the way the previews show.
- [ ] Every built-in preset trains one epoch through the GUI (Load Preset, Start Training) and saves a LoRA.
- [ ] `train_memory` figures are measured, at two resolutions per precision, from the trainer's logged peaks.
- [ ] Pause and Resume: a run paused mid-way and resumed continues from the same epoch, and its loss carries on from where it was.

## Small cards

Most people train on 12–24 GB cards, so a family isn't finished until it has been run on them. Nothing here is optional when the family offers the feature; `FIZGIG_SIM_VRAM_GB=N` makes the planner and the allocator behave as an N GB card, so a large card can run every tier.

- [ ] `train_memory` is measured (above), so Auto has real figures to plan from.
- [ ] **Auto plans fit:** a LoRA run with Auto precision and Auto block swap finishes, previews included, under `FIZGIG_SIM_VRAM_GB=12` and `=16`. Note the plan it logs (`[precision] Auto plan: ...`) and the peak.
- [ ] **Block swap runs**, if the family has it: one run with swap forced on (e.g. bf16 with 10 blocks swapped under a simulated 16 GB card) trains, previews and saves. Built is not the same as working.
- [ ] **Fine-tune plans fit**, if the family fine-tunes: `ft_spec`'s memory figures (`overhead_gb`, `stream_base_gb`, `calib_mp`, `act_gb_per_mp`) come from measured window peaks at two resolutions, and a fine-tune finishes under `FIZGIG_SIM_VRAM_GB=16` and `=24` with the fewest windows that fit.
- [ ] **The workbench renders on a small card:** a Repair Studio render with Auto precision and swap under `FIZGIG_SIM_VRAM_GB=12`.
- [ ] The text encoder fits the smallest card you support on its own (it runs alone while caching); if it doesn't, give it a quantised fallback as Qwen 2.1's does.

## Workbench

- [ ] Each tab named in `workbench` loads the family and renders: Repair Studio, the Explorer (four variants, pick one), the Profiler (a quick profile writes its report), Extract (a rank-reduced file), LoRA Royale (two checkpoints).
- [ ] **Save Repaired is faithful:** a LoRA saved with some blocks switched off renders the same as the live preview it was saved from.
- [ ] With `activation_cache`: a slider change renders the **same picture** with Turbo Preview on as with it off.

## Abilities, if declared

- [ ] Fine-tune (optional; a family can ship without it): a run at learning rate 1e-30 saves a file equal to the source, tensor for tensor (only weights that were exactly zero may move). Then one real run, loaded in ComfyUI in place of the base, and a small-card run under `FIZGIG_SIM_VRAM_GB`. See [FINETUNE.md](FINETUNE.md).
- [ ] Edit training: an edit LoRA trained on a few pairs applies the edit to a new photo.
- [ ] Sliders: at strength −1, 0 and +1 the previews move between the two looks, and 0 matches the base model.
- [ ] Video: an off-spec clip (wrong frame rate or frame count) is refused at Start with the reason; a clip preview plays with sound (if the model has sound); Repair Studio plays the tweaked, baseline and no-LoRA clips.

## What a finished family includes

The checks above prove the family works. These make it good, and every family should have them unless it has a measured reason not to. Most are one field in the description.

**Every ability your model can support.** People expect the same tools on every model, so switch on each ability the model is able to do, not only the one you came for. Most cost a field or two: slider LoRAs (`slider_training`: photo pairs need nothing more, prompt sliders two short methods), LoKR in `network_types`, EMA (`ema_default`), the per-image loss watch, a speed LoRA for fast previews, Turbo Preview in Repair Studio (`activation_cache`), every workbench tab, compile and INT8 attention where they measure faster, edit training if the model takes reference images, and full fine-tuning (which can also come later). Leave one out only when the model can't support it or a measurement shows it hurts, and say why in a comment beside the description. [ABILITIES.md](ABILITIES.md) lists them all.

**Research before presets.** Before writing presets or sampling defaults, find what the model's users actually settled on, not only the model card: the sampler, scheduler, steps and CFG; the learning rate, rank and alpha, epochs or steps per dataset size, and optimizer; and which layers must never be trained (Anima's LLM adapter, for example). Cite the sources in comments beside the values.

**Parity with the reference.** Measure your text conditioning, one model prediction and the sigma schedule against the reference implementation, from identical inputs. If your model code isn't the reference's own code, compare in fp32. A config file written for a newer library can be misread silently; spell such values out in the driver.

**Presets.**
- Use `adamw` (fused AdamW), not `adamw8bit`, for LoRA presets. 8-bit AdamW steps each LoRA tensor separately, and on a model with hundreds of LoRA'd layers that costs a large share of every step (0.3 s per step on SDXL).
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
