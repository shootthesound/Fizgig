# Adding a stills model

This walks through the smallest family that trains LoRAs, previews them and opens in every workbench tab. Qwen Image 2.1 (`families/qwen_image.py`, `qwen_image21/driver.py`) is a complete version of everything below; keep it open while you work. The names here (`mymodel`, `MyModelDriver`) are placeholders.

Start with **`hidden=True`** in the description. The family is then registered (the command-line cache and trainer find it) but not offered in the GUI, so you can build it a piece at a time. Remove it when the [checklist](CHECKLIST.md) passes.

## 1. The model package

Put the model code in `src/fizgig/mymodel/`: the DiT (or UNet), the VAE and the text encoder, plus a `sampling.py` if your sampler needs one. Vendored code is fine; keep its licence header and note the source.

Three rules make it fit the layer:

- **Plain `nn.Linear` layers** for everything a LoRA should reach. The LoRA wraps Linears by dotted name, and INT8 / NF4 quantise the same ones.
- **The text encoder only encodes.** Captioning is shared (Krea 2's Qwen3-VL-4B for every family), so leave out an LM head or vision tower if encoding doesn't need it. It fits smaller cards that way.
- **Files loaded by name must work offline.** A tokenizer, processor or config fetched from a Hugging Face repo by name goes in the description's `helper_files` (the model downloader fetches those up front) and is loaded with the driver's `self.from_pretrained(cls, repo, ...)`, or `self.helper_dir(repo)` for loaders that want a folder (diffusers' `from_single_file(config=...)`). Plain `from_pretrained` asks the Hub on every call, so a fully cached setup still fails offline, or when the Hub rate-limits the machine. `families/offline_check.py --family <key>` proves it (see the [checklist](CHECKLIST.md)).
- **Block swap is optional.** If your blocks sit in an `nn.ModuleList` and your forward can call the shared offloader around each block (`wait_for_block` / `submit_move_blocks_forward`, as Qwen's model does), you get block swap and fine-tune streaming on small cards. Without it the family trains without swap.

## 2. The description

`src/fizgig/families/mymodel.py`:

```python
from fizgig.families.description import FamilyDescription, LoRAFormat, ModelFile, SamplingSettings

MYMODEL = FamilyDescription(
    # identity
    key="mymodel",                    # used by the workbench, presets and the CLI's --family
    arch_id="mymodel10",              # cache file names and LoRA metadata: change it and old caches are ignored
    display_name="My Model 1.0",
    gui_label="My Model 1.0 (experimental)",
    lora_name_suffix="mym10",
    hidden=True,                      # not in the GUI yet

    # one row per file the user points Fizgig at (Preferences rows, Download links, the model downloader)
    model_files=(
        ModelFile("mymodel_dit", "My Model DiT", True, "org/my-model", "my_model_bf16.safetensors", 12.0,
                  "bf16 base for training.", role="dit"),
        ModelFile("mymodel_vae", "My Model VAE", True, "org/my-model", "vae.safetensors", 0.3,
                  "", role="vae"),
        ModelFile("mymodel_text_encoder", "Text encoder", True, "org/my-model", "te.safetensors", 8.0,
                  "Used for caching only, then unloaded.", role="text_encoder"),
    ),

    # latent rules
    latent_channels=16,
    spatial_factor=8,                 # pixels per latent cell
    bucket_step=64,                   # training buckets snap to this (a multiple of spatial_factor)

    # the default block map: n_blocks numbered blocks, each wrapping lora.block_modules
    n_blocks=28,
    block_prefix="blocks",

    # how saved LoRAs are keyed, so ComfyUI loads every module
    lora=LoRAFormat(
        key_template="diffusion_model.blocks.{block}.{module}.{ab}.weight",
        down="lora_down", up="lora_up",
        block_modules=("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"),
        file_prefix="diffusion_model.",
        source="ComfyUI comfy/lora.py, the MyModel branch",
    ),

    driver="fizgig.mymodel.driver:MyModelDriver",
    modelspec_arch="My-Model-1.0",
    implementation="https://github.com/org/my-model",

    sampling=(SamplingSettings("Reference", steps=30, cfg=4.0, source="the model card"),),
    workbench=("repair", "explorer", "profiler", "extract", "royale"),
)
```

Register it in `families/registry.py`: import `MYMODEL` and add it to the tuple in `FAMILIES`. The order there is the Base Model dropdown's order. The registry runs `validate()` on import and refuses an inconsistent description, so a missing field fails loudly, not halfway through a run.

Every value that came from outside Fizgig should carry a `source=` (or a comment naming it), so the next person can re-check it when the upstream model moves on.

### Getting the LoRA format right

Most of a family's "it trains but does nothing in ComfyUI" bugs live here. Before training anything:

1. Find how ComfyUI maps LoRA keys for your model (`comfy/lora.py`, `model_lora_keys_unet`) and copy that key shape into `key_template` and `file_prefix`.
2. Check fused layers. If ComfyUI's checkpoint fuses two Linears into one tensor (Qwen's MLP `gate_up`, Klein's `qkv`), make sure the key prefix you choose is one ComfyUI splits correctly. Qwen's description notes the case where a different prefix silently drops half the MLP.
3. LoRAs written by other trainers (diffusers naming, OneTrainer, AI-Toolkit) load if the driver maps their names: `alias_flat(flat)` for renames, `convert_lora_state_dict(sd)` for layout changes (Klein fuses diffusers' split q/k/v there).

## 3. The driver

`src/fizgig/mymodel/driver.py`. These are the methods with no default; everything else in `FamilyDriver` is optional.

```python
import torch
from fizgig.families.driver import FamilyDriver


class MyModelDriver(FamilyDriver):

    # ---- models
    def load_dit(self, path, device):
        """bf16, frozen. INT8 / NF4 are applied afterwards by the layer, to the block map's Linears."""
        from fizgig.mymodel.model import load_dit
        return load_dit(path, device=device).eval().requires_grad_(False)

    def load_vae(self, path, device): ...
    def load_text_encoder(self, path, device): ...
    def unload_text_encoder(self, te): te.unload()
    def enable_gradient_checkpointing(self, dit, on=True): dit.enable_gradient_checkpointing(on)

    # ---- encoding (the cache calls these)
    @torch.no_grad()
    def encode_images(self, vae, images):
        """uint8 (H, W, 3) arrays, all one size -> list of (C, h, w) latents in the space the DiT trains on
        (normalised, on CPU)."""

    @torch.no_grad()
    def encode_text(self, te, captions):
        """-> one dict of CPU tensors per caption. The keys are yours; the cache stores the dict and hands it back,
        batched (leading dim 1), to training_loss and generate."""
        return [{"hidden_states": h.cpu()} for h in te.encode(captions)]

    # ---- training
    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                      diff_weight=0.0):
        """One forward. latents (1, C, h, w) on the device. Owns the noise schedule, the timestep sampling (scaled
        into [min_t, max_t] - the Training tab's Timestep Range), the prediction and the target.
        Returns (loss, {"t": t})."""

    # ---- sampling (previews and the workbench)
    @torch.no_grad()
    def initial_noise(self, seed, width, height):
        """Exactly the noise generate() starts from for this seed, on CPU (seed travel blends two of these)."""

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None,
                 options=(), noise=None, on_step=None, refs=None):
        """Denoise one image -> latents (your layout). Call on_step(i, steps) before every step and let it raise:
        that is how a render is cancelled, and how Turbo Preview knows which step it's on. noise, when given,
        replaces the seed's start. sigmas / options come from a speed LoRA's settings; ignore what you don't use."""

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        """-> PIL.Image (RGB)."""
```

### The training step

`training_loss` is where a family's identity lives, and the trainer trusts it completely. Match the model's own training recipe:

- **Objective.** Flow matching (Klein, Krea 2, Qwen: `x_t = (1 - t)·x0 + t·noise`, target `noise - x0`) or a DDPM-style epsilon / v-prediction for older models. Nothing outside the driver assumes either.
- **Timestep sampling.** Use the distribution the model was trained with: logit-normal, a resolution-dependent shift, uniform. Then scale it into `[min_t, max_t]` so the Timestep Range control works.
- **Return** `(loss, {"t": t})`, with `t` on the 0–1 scale where 0 is clean. The per-image loss watch buckets by it. You may also return `info["lr_mult"]` to scale this step's learning rate.

### The block map

The default `block_map()` gives one group of `n_blocks` numbered blocks, each wrapping `lora.block_modules`. Override it when your model has named areas (text fusion, refiner, input/middle/output blocks) or modules outside the numbered blocks:

```python
from fizgig.families.driver import Block, BlockGroup

def block_map(self, dit=None):
    blocks = [Block(f"block_{i}", f"Block {i}", [f"blocks.{i}.{m}" for m in self.description.lora.block_modules])
              for i in range(self.description.n_blocks)]
    io = Block("io", "Input / output", ["x_embedder", "final_layer.linear"])
    return [BlockGroup("Blocks", blocks), BlockGroup("Input / output", [io])]
```

Block ids are permanent: presets, Profiler sidecars, saved states and Repair Studio configs refer to them. Labels are what users see, so use the model's own terms. **`block_map()` must work with `dit=None`**: the Profiler and Extract read LoRA files without loading a model.

### Models that don't look like Qwen

Two shipped families cover the common departures. Read the one closest to your model.

**SDXL** (`families/sdxl.py`, `sdxl/driver.py`) is a UNet in one checkpoint file, built from diffusers' own classes:

- **One file holds every part.** The checkpoint carries the UNet, VAE and both text encoders. Its VAE and text-encoder rows set `inside="sdxl_checkpoint"`: left empty, they resolve to the checkpoint (`description.model_path(role, lookup)`), and a user can still point one at a separate file. Each `load_*` method gets the checkpoint path and reads its own part.
- **Named blocks, not numbered ones.** `block_map()` lists the 11 attention modules under the names SDXL tools use (IN04 … MID … OUT05) in three groups. The resnets carry no Linears, so they're left out.
- **Other trainers' names.** Most community SDXL LoRAs use the original LDM layout (`lora_unet_input_blocks_4_1_…`). `alias_flat` maps those prefixes onto the diffusers names the model uses, so Repair Studio, the Profiler and Extract read them. Their text-encoder parts (`lora_te1_` / `lora_te2_`) are ignored.
- **DDPM, not flow.** `training_loss` uses epsilon prediction on SDXL's scaled-linear schedule, plus the size conditioning (`time_ids`) SDXL needs in every forward. Nothing outside the driver knows.

**Anima** (`families/anima.py`, `anima/driver.py`) is a DiT whose text path runs inside the model:

- **An adapter inside the DiT.** Anima's LLM adapter turns the Qwen3 hidden states and the caption's T5 token ids into the DiT's context. The cache stores both (`{"qwen", "t5_ids"}`), and the driver runs the adapter at the start of every forward instead of caching its output. That way a frozen LoRA that reaches the adapter (Anima's Turbo LoRA does) takes effect. The adapter isn't in the block map, so it's never trained.
- **Variable-length conditioning.** Prompts and negatives encode to different lengths, so `generate` runs CFG as two forwards rather than one stacked batch.
- **Follow the reference inference path exactly.** The driver copies ComfyUI's Anima path: no prompt template, the adapter run on the unpadded sequence, the result zero-padded to 512 tokens. Anything else trains a LoRA for conditioning ComfyUI never produces.
- **Spell configs out in the driver.** Anima's text-encoder `config.json` was written by a newer transformers. Fizgig's pinned version read it without an error, but used the wrong RoPE base: the text encoding drifted further from ComfyUI's with each token, and nothing failed. The driver now lists the encoder's config values itself.

## 4. Memory: let Auto choose

Fill `precisions` (`"bf16"`, `"int8"`, `"nf4"`) and `train_memory` with measured figures, and the Training tab's Auto plan picks the precision and block swap from the user's free VRAM at launch:

```python
precisions=("bf16", "int8", "nf4"),
train_memory={
    # precision: (peak GB with no block swap, GB saved per swapped block)
    # the peak may be ((megapixels, GB), ...) points, interpolated for the run's resolution
    "bf16": (((0.5, 16.0), (1.0, 19.0)), 0.45),
    "int8": (((0.5, 9.5), (1.0, 11.9)), 0.24),
},
```

Measure them with short training runs at two resolutions. The trainer logs the peak each epoch. With `train_memory` empty, Auto takes the first precision and doesn't swap.

Then run the plans on small cards: `FIZGIG_SIM_VRAM_GB=12` (or 16, 24) makes the planner and the allocator behave as that card. The [checklist](CHECKLIST.md#small-cards) lists the runs every family needs: Auto LoRA plans at 12 and 16 GB, block swap forced on, fine-tune plans at 16 and 24 GB, and a workbench render at 12 GB.

## 5. Presets and samples

- **`presets`**: built-in Training-tab presets, `((name, {setting key: value}), ...)`. The first is applied on a user's first visit to the family. Copy Qwen's `_preset()` helper to start, then tune it on real runs. Ship only presets you've trained.
- **`preview_steps`, `preview_cfg`, `preview_width`, `preview_height`**: the Samples tab's defaults for in-training previews.
- **`preview_negative`**: the Samples tab's default negative prompt for the family. Each family keeps the user's own edit. Leave it `None` if the family's previews take no negative (CFG-free or distilled); the box then greys out.
- **`speed_loras` + `preview_speed_lora`**: a Turbo / Lightning / DMD LoRA with the settings it wants (steps, CFG, sigmas). Previews and the workbench use it when its file is set in Preferences, which makes the workbench fast on a slow model. See [ABILITIES.md](ABILITIES.md#fast-previews).

## 6. Run it from the command line

With `hidden=True` the GUI doesn't show the family yet, so cache and train from the command line:

```
python src/fizgig/families/cache.py --family mymodel --stage latents --dataset_config ds.toml --model <vae>
python src/fizgig/families/cache.py --family mymodel --stage text    --dataset_config ds.toml --model <te>
python src/fizgig/families/train.py --family mymodel --dit <dit> --dataset_config ds.toml \
    --output_dir out --output_name test --network_dim 8 --network_alpha 8 --learning_rate 1e-4 \
    --max_train_epochs 2 --sample_prompts prompts.txt --sample_every_n_epochs 1 \
    --vae <vae> --text_encoder <te>
```

The dataset TOML is the same format every family uses ([CLI.md](../CLI.md)). When previews look right and the saved LoRA works in ComfyUI, go through the [checklist](CHECKLIST.md), then remove `hidden=True`. The family then appears in the GUI with its Preferences section, Training and Samples tabs, and every workbench tab named in `workbench`.

## Next

- [ABILITIES.md](ABILITIES.md): switch on edit training, sliders, fine-tuning, Turbo Preview, compile and more, one field at a time.
- [FINETUNE.md](FINETUNE.md): full fine-tuning, if you want it now. It's optional and can come later.
- [VIDEO.md](VIDEO.md): if your model also trains on clips.
