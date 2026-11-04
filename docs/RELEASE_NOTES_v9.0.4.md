# Fizgig v9.0.4: Fibo 2.0

**Fibo 2.0 is a new model in Fizgig** (thanks @spaghettified26, #250). BRIA's 8.2B model for text-to-image and editing trains here as LoRAs, edit LoRAs, slider LoRAs and full fine-tunes, from a 12 GB card. Training previews render in 4 steps with Comfy-Org's Turbo LoRA. Masked training, Studio with Turbo Preview, LoRA Royale with seed and prompt travel, the Profiler and Extract all work on it. Every file Fizgig saves for Fibo, fine-tunes included, loads in ComfyUI as it is.

To set it up, open Fibo 2.0 on the Preferences tab and press **Download models for me**. It fetches about 28 GB from Comfy-Org, with no Hugging Face token needed. Then pick Fibo 2.0 on the Training tab and load a preset. The LoRA preset uses Automagic from 1e-6 for 40 epochs, which reached likeness sooner than AdamW in our runs. Base precision Auto trains INT8 on a 16 GB card and NF4 on a 12 GB card; the int8 DiT build ComfyUI's Fibo workflows use is offered beside the bf16 one. Fibo 2.0's weights are for non-commercial use only; see the [Fibo 2.0 guide](docs/FIBO2.md) for the licence, memory figures and the rest.

**Also new**
- The Training tab is tidier: a preset's note sits after Save preset, and a model's note shows once, under the model picker.
- The guide for adding a model ([docs/drivers/CHECKLIST.md](docs/drivers/CHECKLIST.md)) now spells out what a finished model includes: every feature the model can support, files that download without a token, every saved file loading in ComfyUI, and nothing the other models don't have.

**Fixes**
- MiniMax H3: a Context LoRA from another trainer that adapts the audio or video input and output layers crashed the run on its first step. It now trains (thanks @spaghettified26, #298). A Context LoRA that matches nothing in the model now stops the run with a message, instead of training as if it weren't there.

**Updating:** update Fizgig on your computers, and restart your pods.
