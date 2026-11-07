# Klein 9B

[← Back to the README](../README.md)

Flux 2 Klein 9B is Fizgig's fast, light trainer, tuned for one model. Training runs on the **Base** DiT; the **Distilled** DiT powers the 4-step previews and the workbench tools.

## Training

- **Proven presets** for single subject through multi-character — or roll your own.
- **Edit LoRAs** — pairs of an original and your edited version (Training Parameters → Edit LoRA). Pick **The photo + extra pictures** to bring in up to 3 more pictures per edit (another person, an object, a place) from their own folder, named `photo_1.png`, `photo_2.png` … for `photo.png`, any shape.
- **Context LoRA** — load an existing LoRA as a frozen *active* layer so the new one learns to coexist: a face on top of a style, an outfit on top of a character. No other trainer does this.
- **Adaptive LR** — a bi-directional plateau tracker: set the Min/Max window and it probes up on steady descent, pulls down (with rollback) on plateau or instability. A rate that had to be cut soon after a raise becomes that run's ceiling, so it isn't tried again.
- **fp8 Base training** — the fp8 Base stays resident at ~9.6 GB, so a full 9B LoRA trains in ~14 GB and fits a 16 GB card. Automatic.
- **Distilled training samples** — 4-step previews that match ComfyUI output closely, multiple prompts (one per line on the Samples tab), and optional **reference-conditioned** samples (Klein is an edit model — previews can edit a real photo).
- **Pause / Resume** — graceful epoch-boundary pause that frees your GPU mid-run and resumes with full state.
- **Model Area targeting** — train only Identity, Style, or Detail blocks, or the full model.
- **Per-dataset caches, cross-checked** — deleted images leave the run; switched datasets can never leak in.
- **Bilingual captions** (English + Chinese via Helsinki-NLP) act as text-level augmentation — measurably better skin detail on Klein at identical loss. See [Dataset prep](TRAINING.md#dataset-prep).

The sample gallery, dataset prep and LoRA format support are shared with the other models: see [Training features and the workbench](TRAINING.md).

## Block map

Klein is the model with a charted block map, which drives Model Area targeting, the Profiler's colour-coded report, the colour-coded Repair Studio sliders, and the Extract tab's targeted presets.

| Category | Blocks |
|---|---|
| Style + Composition | double blocks 0-7, single blocks 0-1 (and single 2 at half weight when extracting) |
| Identity | single blocks 1-16 |
| Details | single blocks 12-23 |

The overlaps are real, so the Profiler reports five buckets (style+composition, style/identity overlap, identity, identity/detail overlap, details) rather than forcing each block into one. Style also concentrates at **late timesteps** (trainer `min_timestep=0, max_timestep=400`), so the Style area pairs the style+composition blocks with that range.

On the workbench, Klein's Repair Studio has a slider for each of its 32 blocks (8 double + 24 single), and Extract's presets keep one part of a LoRA — Identity, Style+Composition or Details — or any blocks you pick. Full tool descriptions are in [TRAINING.md](TRAINING.md#the-workbench).

## Model files

You only need these if you train Klein. **Preferences → ⬇ Download models for me** under the Klein card downloads, verifies and fills in the paths; Klein needs a free HuggingFace token for BFL's licence. Every row also has a manual **Download** link. From the command line (~34 GB, needs a token):

```bash
python -m fizgig.scripts.fetch_models --family klein
```

| Model | File | Size | Source |
|---|---|---|---|
| **Base DiT (fp8) — recommended** | `flux-2-klein-base-9b-fp8.safetensors` | ~9.5 GB | [black-forest-labs/FLUX.2-klein-base-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8) |
| Base DiT (bf16) | `flux-2-klein-base-9b.safetensors` | ~17 GB | [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) |
| Distilled DiT | `flux-2-klein-9b-fp8.safetensors` | ~9 GB | [black-forest-labs/FLUX.2-klein-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8) |
| VAE / AE | `ae.safetensors` | ~320 MB | [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/ae.safetensors) (from root, **not** the `vae/` subfolder) |
| Text Encoder | `qwen_3_8b.safetensors` | ~15 GB | [Comfy-Org/vae-text-encorder-for-flux-klein-9b](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/main/split_files/text_encoders/qwen_3_8b.safetensors) |

Train on the **Base DiT** — the fp8 version is recommended on every GPU (same quality, half the VRAM). The VAE must be `ae.safetensors` from the root of the FLUX.2-dev repo; the Diffusers-format file in its `vae/` subfolder is the wrong one.

General install steps and the other models' files are in [INSTALL.md](INSTALL.md).

## VRAM

**Training** — the fp8 Base stays resident at ~9.6 GB, so a 9B LoRA fits **16 GB** (~14 GB observed). Smaller cards: the **4-bit (NF4) base** (Training tab → Base precision, which Auto picks on cards under 16 GB) drops the base to ~5.6 GB — a full LoRA trains in about 8.5 GB at 0.5 MP, fitting **10–12 GB cards with no swap**.

**Workbench** (Distilled 4-step):

| Block Swap | Min VRAM |
|---|---|
| 0 | 16 GB+ with the fp8 Distilled model, 24 GB+ with bf16 |
| 8 | 16 GB (bf16) |
| 12 | 14 GB |
| 16 | 12 GB |

On first launch Fizgig detects your VRAM and picks the default (0 on 16 GB cards and up); your own choice sticks.
