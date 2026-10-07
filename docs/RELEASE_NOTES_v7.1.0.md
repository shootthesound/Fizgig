# Fizgig v7.1.0: Z-Image Turbo, trained right

Z-Image Turbo deserved a longer run as a model to train. It arrived as one of the best photo models anyone could run at home: fast, sharp, 8 steps, no CFG, and it is still one of the most-used image models in ComfyUI. Training it was another story. A LoRA trained on Turbo the usual way undoes the distillation that makes it fast, and the pictures go soft and hazy. The adapter most people trained with held it together, but left glossy, over-processed skin and pulled the rest of the model with it. Plenty of people never got it to train the way they wanted, and when newer models arrived, most of the training moved with them.

Fizgig v7.1.0 makes it worth training again. Z-Image Turbo joins as a full model family with its own training adapter, built and tested in Fizgig, that keeps Turbo looking like Turbo while your LoRA learns.

Fizgig trains on Turbo itself, the model you generate with. We're open to adding training on Z-Image Base, but in our tests Turbo now responds very well to training.

## Headlines

- **Z-Image Turbo is in Fizgig,** with LoRA and LoKR training, sliders, full fine-tuning and every workbench tab: Repair Studio, LoRA the Explorer, Profiler, Extract and LoRA Royale.
- **Fizgig's own training adapter for Z-Image Turbo.** It sits frozen under your LoRA while it trains, is off in previews and never in your saved file, and keeps Turbo's 8-step look intact. Against the adapter most people use, it gives better likeness, better image quality and better detail: clean, natural skin instead of gloss and speckle, and prompts that don't mention your subject stay looking like Turbo.
- **Train on the card you have.** LoRAs train from 12 GB cards, and full fine-tunes from 16 GB. The Auto plan picks the precision and block swap that fit your card.
- **Your LoRAs work in ComfyUI** with the normal LoRA loader on Z-Image Turbo, at 8 steps and CFG 1.
- **Works offline.** Once its files are downloaded, Z-Image trains and previews with no internet connection.

## Getting started with Z-Image Turbo

1. Pick **Z-Image Turbo (experimental)** in the Base Model picker.
2. On the Preferences tab, download the four files in the Z-Image Turbo section: the Turbo model (12.3 GB), the Qwen3-4B text encoder (8 GB), the FLUX.1 VAE and the Fizgig training adapter (70 MB). If you already use Z-Image in ComfyUI you can point the rows at your own files instead; the VAE is the FLUX.1 one (often saved as `ae.safetensors` or `z_image_vae.safetensors`), not the FLUX.2 VAE Klein uses.
3. Load a preset and train. The training adapter is ticked by default; leave it on.

Previews render on Turbo itself at 8 steps with no CFG, so the Samples tab hides CFG and the negative prompt for Z-Image.

## Presets

The presets follow Qwen Image 2.1's. Fast trains at 0.25 MP, the rest at 0.5 MP:

| Preset | For |
|---|---|
| ✨ Z-Image Turbo Fast (rank 8, adaptive LR) | Characters and most LoRAs, quickest to a likeness |
| ✨ Z-Image Turbo Standard (rank 16, adaptive LR) | Bigger or more varied datasets |
| ✨ Z-Image Turbo Style (rank 16, 1.5e-4) | Styles |
| ✨ Z-Image Turbo Slider (rank 8, 2e-4) | Slider LoRAs, from photo pairs or prompts |
| ✨ Z-Image Turbo Fine-tune (5e-5) | A full fine-tune |
| ✨ Z-Image Turbo Fine-tune Strong (1e-4) | A fine-tune that should move the model further, such as a strong style |

Z-Image fine-tunes at 5e-5 rather than the 1e-5 the other models use: its weights are larger, so 1e-5 hardly moves it. Ticking Fine-tune on the Training tab sets 5e-5 for you. Every epoch saves (every rotation for a fine-tune), so you can pick the best one in LoRA Royale.

## Z-Image on smaller cards

Measured with the Auto plan, previews included:

| Your card | LoRA training | Full fine-tune |
|---|---|---|
| 12 GB | INT8 with block swap | — |
| 16 GB | INT8, no block swap | three parts of the model at a time |
| 24 GB | bf16, no block swap | two parts at a time |
| 32 GB | bf16 | the whole model at once |

Repair Studio and the other workbench tabs run on a 12 GB card.

## Also in this release

- **ReFLoRAs in RefMod Studio.** Open a plain MiniMax H3 LoRA or a ReFLoRA beside your RefMods, compare against no mod or no LoRA, and save your LoRA and active RefMods together as a ReFLoRA, in the H3 RefLoRA v1 format. Suggested by @NftGamer666 (#177); the format is @malcolmamal's.
- **RefMod Studio is clearer.** A card at the top says what the tab does, the LoRA has its own card that says what's loaded, a ReFLoRA's RefMods appear as the Mods rows, and the comparison and Save as ReFLoRA only show when a LoRA is set.
- **AMD ROCm: no more streaks in previews.** The VAE's wide attention heads returned wrong values on ROCm builds, which showed as horizontal streaks in previews and cached latents (Krea 2, Anima, Qwen 2.1, Klein, SDXL). Fizgig now computes them directly on ROCm. NVIDIA cards are unchanged. Reported with a diagnosis by @ArchAngelAries (#179). On AMD, update with `update_fizgig_rocm.bat`.
- **Adding a model to Fizgig:** the driver guide gains a Small cards checklist, a requirement that a model works offline once its files are cached (with `families/offline_check.py` to prove it), and a page of real code from the Qwen Image 2.1 and MiniMax H3 drivers, kept identical to the source.
- **Slider presets train at rank 8** on Qwen Image 2.1, Krea 2 and Anima (they were rank 4), as on MiniMax H3 and Z-Image.
- **Repair Studio bulk buttons on every model.** Reset all, All off, All on and Invert now sit above the per-block sliders for every model, not just MiniMax H3 (which keeps Alternate and Toggle detail blocks). While a donor is loaded, Donor: all on and Donor: all off join the end of the same row.
- **A donor LoRA is on when it loads.** Its sliders start at 1 on every block (they started at 0, so a newly loaded donor changed nothing until you moved them). Swapping one donor for another still keeps the sliders where you left them.
- **Browse opens where your LoRA is.** Every LoRA Browse button (Repair Studio primary and donor, LoRA the Explorer, Profiler, Extract, Metadata, Context LoRA) now opens in the folder of the LoRA already in its box. An empty donor box opens in the primary's folder.
- **A tip on the Samples tab:** give one sample prompt a pose, angle or setting your photos don't have. When a LoRA starts to overtrain, that is the first thing its previews stop following, so the last epoch that still follows it is a good one to keep.
- **The Samples tab's Sample Output line** now follows the model you pick, instead of showing the previous model's folder.
- **Notices:** the MiniMax H3 code derived from ComfyUI is marked GPL-3.0, and the Z-Image code adapted from Tongyi-MAI's reference implementation is credited under Apache-2.0.
- **GitHub Sponsors:** the repo has a Sponsor button, beside Buy me a coffee.

## Credits

Thank you to @ArchAngelAries for tracking down the ROCm streaks, @NftGamer666 for the ReFLoRA idea, and @malcolmamal for the RefLoRA format.
