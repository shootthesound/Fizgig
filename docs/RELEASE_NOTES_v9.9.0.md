# Fizgig v9.9.0: Style presets that learn the style, and CAME is back

**Style presets now learn the style.** Most models' Style preset now trains on **CAME at a fixed 5e-5**. We trained each one on the same 8-picture papercut style and rendered prompts the set never showed. On MOSS-Image, the Style preset on Muon at 5e-5 learned no style in 50 epochs: every picture came out a photo. CAME at the same rate had a clean papercut look by epoch 20. CAME learned the style on every model below. The change covers MOSS-Image, Anima, Qwen Image 2.1, Flux 3, Krea 2, Z-Image Turbo, MiniMax H3 Image and MiniMax H3 video. On both H3 models, which trained on Automagic before, the style also reached people sooner: at epoch 20, Automagic's portraits were still nearly photos, and CAME's were in the style. Klein and Z-Image Base keep their Style presets as they were (CAME at 5e-5 learned no style on Z-Image Base). Rank, epochs and size are unchanged; the learning rate is in each new name, such as **Qwen 2.1 Style (rank 16, 5e-5)**. Load the Style preset again to use it.

**SDXL has a Style preset.** **SDXL Style (rank 16, alpha 8, 5e-5)** is SDXL Standard on CAME, and had the papercut style by epoch 20 in the same test.

**CAME is back in Optimizer Type, for every model.** It's an Adafactor variant with Adafactor's small memory use, whose steps grow where the update holds steady, so at the same learning rate they run larger than AdamW's. The Learning Rate box, Adaptive LR and the scheduler drive it as they drive AdamW. Fizgig carries its own copy of the official implementation (MIT), so there's nothing extra to install. (#416, thanks @sam00778)

**Studio says why a save failed.** When Save repaired LoRA failed, the save card said "Saving…" for ever; it now shows "Couldn't save:" and the reason. (#413, thanks @Aymenbayoss)

**A smoother VAE for H3 Image in ComfyUI.** The H3 Image guide now points to ostris's fine-tune of the H3 Image VAE's decoder, which takes the over-sharp edge off skin, hair and small text. It works with any H3 Image LoRA, and you can set it in Preferences to see the same look in Fizgig.

**To update:** run the updater as usual.
