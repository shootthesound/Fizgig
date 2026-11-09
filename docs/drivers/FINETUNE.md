# Fine-tuning a family

**Fine-tuning is optional.** A new family is complete without it: LoRA training, previews and every workbench tab work on their own, and nothing else depends on fine-tuning. Send your family without it if you like. You, the maintainer or anyone else can add it later in its own change, by following this page.

Full fine-tuning trains the base model's own weights instead of a LoRA, and saves a full checkpoint in the source file's layout, so ComfyUI loads it in place of the base. The shared code in `src/fizgig/families/ft.py` does the work. A family supplies one method that says which weights to train, plus some memory figures you measure once.

## How it runs

- The frozen base sits on the GPU in 4-bit (an NF4 trunk). The weights being trained are rebuilt in bf16 from a bf16 master copy read from the model file, never from the 4-bit copy.
- The trained weights are grouped into **parts**: Linear-name prefixes inside each block, such as attention and the MLP matrices. A **window** is the set of parts training at the moment, always across the model's full depth unless the card forces a split.
- The planner fits the card:
  1. **Every part in one window** when the card holds the whole model. Nothing rotates.
  2. Otherwise, **as few windows as fit**, several parts per window where they fit together.
  3. Otherwise, one part per window, **split by depth** where a part doesn't fit whole.
  4. Otherwise, **streaming**: the blocks outside the window stay in system RAM and stream to the GPU as the forward reaches them.
- With more than one window, a rotation trains each window in turn for `rotate every` epochs. The Training tab shows the plan for the user's card before Start, and the log confirms it.
- The Training tab's **Window size** setting caps how many parts share a window (at most 3, at most 2, or one each), for a user who wants more headroom than Auto leaves. It reaches your planner as `max_parts`.
- The optimizer is Adafactor by default, with each gradient applied and freed as it lands, so a window costs about its own bf16 weights. The user can pick AdamW8bit, AdamW, AdEMAMix8bit or Lion8bit instead (their larger state is added to each window's plan, `optimizers.FT_OPTIMIZERS`); a LoRA optimizer (Muon, Prodigy, Automagic) trains as Adafactor. A family with its own fine-tune backend keeps Adafactor unless its backend sets `plans_optimizer_state`.
- Checkpoints are written one tensor at a time: the untouched tensors are copied from the source file, the trained ones come from the master with the live GPU weights laid over it.

## What your model needs

- **A 16- or 32-bit model file.** The default `ft_source_unfit` refuses fp8, int8 or pre-quantised files: they have no full-precision layout to build the master from or to write the checkpoint into. The master keeps the file's own precision (bf16, fp16 or fp32), and only what training changed is written back, so an fp16 checkpoint's untouched weights come back exact.
- **Linears that quantise to NF4.** The trained parts must be among `quant_target_names(dit)`, which defaults to the LoRA targets. Only NF4 Linears rotate.
- **Blocks in an `nn.ModuleList`**, or several lists that count as one cycle (Klein: 8 double blocks, then 24 single blocks).

## The two steps

### 1. The description

```python
finetune=True,
ft_learning_rate=1e-5,     # the rate choosing Fine-tune sets; 1e-5 unless the model's users found better
```

### 2. `ft_spec` in the driver

```python
def ft_spec(self, dit):
    from fizgig.families.ft import FTSpec
    return FTSpec(blocks="blocks", components=("self_attn", "cross_attn", "mlp.layer1", "mlp.layer2"),
                  file_prefix="net.", overhead_gb=2.5, calib_mp=1.0, act_gb_per_mp=2.0)
```

That is Anima's. `dit` may be `None`: the Training tab calls `ft_spec` before any model loads.

| Field | What to put |
|---|---|
| `blocks` | The attribute holding the numbered blocks, or a tuple of attributes numbered one after another (Klein's `("double_blocks", "single_blocks")`). |
| `components` | The parts, as Linear-name prefixes within a block, in rotation order. Make them similar in size: the biggest part sets the smallest card that can hold a full-depth window. In a model with several block lists, a part may exist in some blocks only; the planner splits it only across the blocks that have it. |
| `always_on` | Modules outside the blocks trained for the whole run (Krea 2's text fusion). Their Linears must stay bf16, outside `quant_target_names`. |
| `file_layout` | When your model splits or renames a weight the file stores differently. Qwen's file fuses the MLP gate and projection into one `gate_up` tensor: `(("img_mlp.gate_layer.weight", "img_mlp.gate_up.weight", 0, 2), ("img_mlp.proj.weight", "img_mlp.gate_up.weight", 1, 2))`. |
| `file_prefix` | A prefix every DiT key in the file carries (Anima's `net.`, a single-file SDXL checkpoint's `model.diffusion_model.`). A `diffusion_model.` prefix is found without it. Keys without the prefix (a checkpoint's text encoders and VAE) are copied into the saved file untouched. |
| `file_names` | Where the file names a block differently from the loaded model: `(model prefix, file prefix)` pairs. SDXL loads in diffusers names but its checkpoints use the LDM layout, so `("down_blocks.1.attentions.0.", "input_blocks.4.1.")` and so on. |
| `overhead_gb`, `calib_mp`, `act_gb_per_mp` | The planner's memory figures; see [Measure the memory](#measure-the-memory). Until you measure, the cautious defaults apply, and they refuse cards that could run it. |
| `stream_base_gb`, `trunk_gb_per_block` | Streaming figures; leave them unset unless streaming plans come out badly. |

**Leave out what must never be trained.** Check what the model's users found: Anima's LLM adapter degrades if trained, so it's not a part. Modulation layers, embedders and the final layer usually stay frozen too.

## Streaming on small cards

Streaming uses the same interface as block swap: your forward calls `offloader.wait_for_block(i)` before block `i` and `offloader.submit_move_blocks_forward(blocks, i)` after it. With one block list and the usual `offloader` + `blocks_to_swap` attributes, the default `install_ft_streamer` plugs the streamer in.

A model with several block lists, each swapped by its own offloader counting from 0, gives each list a view of the one streamer:

```python
def install_ft_streamer(self, dit, streamer):
    dit.disable_block_swap()
    nd = len(dit.double_blocks)
    dit.offloader_double = streamer.view(0, nd)
    dit.offloader_single = streamer.view(nd, len(dit.single_blocks))
    dit.blocks_to_swap = 1
```

That is Klein's. A model with no block swap can still fine-tune: it simply needs a card that holds its smallest window.

A model whose blocks sit in several small lists (SDXL: 11 attention modules of 2 or 10 transformer blocks) lists each one in `blocks`; a part is then sized and split over the lists that hold it.

## Measure the memory

1. Train one rotation at 1 MP with `rotate every` at 1, so each epoch is one window. The log prints each epoch's `peak VRAM` and, before the first epoch, `planning with X GB allocated`.
2. For each window: `peak − X + trunk − the window's bf16 GB`. The trunk is the 4-bit size of all trained Linears (about 0.27 bytes per bf16 byte). The largest result, plus a little slack, is `overhead_gb`, with `calib_mp=1.0`.
3. Repeat at 0.25 MP. How much the peaks grow per extra megapixel gives `act_gb_per_mp`.
4. Check smaller cards with `FIZGIG_SIM_VRAM_GB=8` (or 12, 16 or 24): the planner and the allocator then behave as that card would. The run must finish, and the plan should be the fewest windows that fit.

## Your own file format or trunk

Two hooks cover families whose files don't follow the shared bf16 + NF4 path:

- `ft_backend(dit, device, src, group)` returns your own backend: MiniMax H3's reads an int8 file and saves back into it.
- `ft_card_plan(path, free_gb, mp, options, max_parts)` returns the Training tab's plan for such a backend.

Both can use the shared window planner, `plan_component_windows` in `krea2/rotation.py`, with their own figures. A window of several parts is credited the 4-bit copies its layers drop when they train (`trunk_credit`); if your backend's measured peaks come out above that, turn the credit off and add `pack_margin_gb`, as H3 does.

## Checks before you ship it

- A run at learning rate 1e-30 (the Training tab refuses 0) saves a file equal to the source, tensor for tensor. Only weights that were exactly zero may move.
- One real run, whose checkpoint loads in ComfyUI in place of the base and shows the training.
- A small-card run under `FIZGIG_SIM_VRAM_GB` that finishes.

Examples: `qwen_image21/driver.py` (`file_layout`), `anima/driver.py` (`file_prefix`, measured figures), `klein/driver.py` (two block lists, streamer views), `krea2/driver.py` (`always_on`), `minimax/driver.py` (its own backend).
