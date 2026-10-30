# Fizgig v8.9.94: a new MOSS-Image Clean-end schedule, and prompt sets

**MOSS-Image: a new Clean-end share default for LoRAs and sliders, `60:7:30:8:3`.** 60% until epoch 7, 30% at epoch 8, then 60% with every 3rd epoch at 30%. On a 53-photo dataset it held likeness about 5 points higher from epoch 9 to 20 than the old default, which slid from 60 to 30 over epochs 8 to 15; the slow slide's 30–40% epochs cost the most. Your saved settings keep the value you set. [Clean-end share](https://github.com/shootthesound/Fizgig/blob/master/docs/MOSS_IMAGE.md)

**MOSS-Image Slide: how the alternating epochs work now.** With **then every Nth epoch** ticked, after the slide the run trains at the higher of the two shares, with every Nth epoch at the lower one, whichever share you put first. The box sets how often the lower share comes round, and those epochs train at **half the learning rate**: they still settle composition but cost less of the likeness the other epochs build. The Training tab reads *then every 3rd epoch at 30%*.

**MOSS-Image fine-tunes get their own default, `11:4:30:8:2`.** The model's own schedule (about 11%) until rotation 4, so body shape and what the model already knows hold, sliding to 30% by rotation 8, then 30% with every 2nd rotation at 11%. At the LoRA's 60%, a plain "a woman" prompt had turned into the trained person by epoch 10, even with DOP. Picking **Fine-tune** on the card sets it, and picking a LoRA again puts the LoRA's back (thanks @RitschRatsch666, #223).

**Prompt sets.** Under the Samples tab's prompts, **Save as set…** keeps your prompts under a name and **Load** puts a saved set in their place. Sets are shared by every model, so a model you train for the first time can start with your usual prompts. With **Change the previews of the run going now** ticked, a loaded set goes to the run training now from its next preview (thanks @VRAM-Hoarder, #267).

**LoRA Royale likeness from up to 3 photos.** Choose up to 3 subject photos from a thumbnail grid, from any folder, and each epoch scores the mean against them, as the Samples gallery does (thanks @zer0signa1, #263).

**Fixes**

- **MOSS-Image fine-tunes: a Split or Slide Clean-end share counted epochs, not rotations,** so each window in a rotation trained at a different share and the schedule moved with the card. It counts rotations now.
- **MiniMax H3 reference distillation (Learn identity from my dataset) crashed on every picture** with a `mat1 and mat2` shape error. It trains again (thanks @gradyhouston, #264).
- **✕ on a training folder in Your training data didn't keep it off the list** when a past run had trained on it (a NAS folder never left). It stays off now until you pick it again; the files are never touched (thanks @amoszach, #266).
- **Continue on a run's card left the Training tab showing the settings from before**, so a continued run could look as if it had the last run's settings (a masked run looked unmasked). The tab now shows the run that's training.

**Updating:** update Fizgig on your computers, and restart your pods.
