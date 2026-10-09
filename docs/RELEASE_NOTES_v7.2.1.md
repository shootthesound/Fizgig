# Fizgig v7.2.1: maintenance

Fixes from the first hours of 7.2, plus model downloads that fetch everything the default setup uses.

## Fixes

- **Samples tab: your CFG and Steps stay as you set them.** On Qwen Image 2.1 with the Turbo LoRA at strength 0, a CFG of 3 came back as 1 after switching models. Steps and CFG now switch to the Turbo's defaults only the first time you open a model, or when you turn the Turbo on or off (a strength above 0 counts as on); anything you've set for a model stays. Leaving the Turbo strength box applies the switch straight away. Reported by @mabseyuk.
- **RefMod Studio: Render Sweep uses the LoRA on the tab.** It used whichever LoRA was last rendered, or none if you went straight to a sweep. Reported by @mabseyuk.
- **RefMod Studio: the clip players name the comparison you've set** (LoRA alone, Mods alone, Neither) instead of always "No mod". Reported by @mabseyuk.

## Model downloads

- **Fizgig's training adapters for Qwen Image 2.1 and Z-Image Turbo download by default.** They were only fetched with the optional tick, although training without them gives worse LoRAs. If you set up either model before this release, press **Download models for me** in its Preferences section to fetch the adapter.
- **Krea 2's Turbo LoRA downloads by default.** Training previews use it. The Turbo DiT stays an optional download.

## Model names

- **No model is called "experimental" in the Base Model picker or Preferences any more:** Qwen Image 2.1, SDXL (any checkpoint), Anima and Z-Image Turbo in the picker, Krea 2, Qwen Image 2.1 and MiniMax H3 in Preferences, and Krea 2's Text fusion presets in Repair Studio. Your saved settings for each model (output folder, Samples settings, negative prompt, saved presets) carry over to the new names.

To update, run `update_fizgig.bat` (`update_fizgig_rocm.bat` on AMD). Pods pick up the update on their next start.
