# Fizgig v9.2.2: quick MOSS-Image previews that look right, and full-size previews on 16 GB cards

**MOSS-Image previews are quick by default and still a good read of your LoRA.** Training previews now render on the Base with the Turbo LoRA at 0.6, 8 steps, CFG 2 and the SFT look LoRA. The Samples tab has a **Use the MOSS-Image Turbo LoRA for previews** tick (on) with its Strength beside it; untick it for full-quality previews. The Turbo and SFT look LoRAs download with the MOSS-Image models, and the updater fetches them for installs that have MOSS-Image set up. Studio, LoRA Royale and the Profiler follow the Samples tab, so they render with the same recipe, or on SFT when its tick is on.

**Fixed**
- **Previews on cards under 20 GB keep their full size** (#309). A memory saver meant only for Krea 2 shrank every model's training previews to 768 px on the long side, so a 768×1024 MOSS-Image preview rendered at 576×768 and looked soft, oddly coloured and broken in the face. Now only Krea 2 caps them. Tested on a 16 GB card: MOSS-Image previews at 1024×1024 peak at 9.5 GB. Thanks to @arkinson9 for the report.
- **Turbo previews use the right strength.** MOSS-Image and Krea 2 training previews ran the Turbo LoRA at 1.0 instead of their recommended 0.6 and 0.7.

**Also**
- **Pods resume faster with compile on:** the compiled blocks are kept on the pod's volume, so a Resume after a restart, Pause or Move reuses them instead of compiling from nothing.
- **The warm-up note is measured:** it appears only when the first steps are actually slow, says how fast they are now against the start, and ends with "Up to speed".

**Updating:** run the updater as usual, and restart your pods. If you're coming from 9.0.x, see the [9.2.1 notes](https://github.com/shootthesound/Fizgig/releases/tag/v9.2.1): Fizgig needs Python 3.11 or newer and an NVIDIA driver with CUDA 13.
