# Fizgig v9.8.2: MiniMax H3 Image fine-tuning, and Studio as a generator

**MiniMax H3 Image fine-tunes.** The Training tab's kind of training now offers **Fine-tune** for MiniMax H3 Image, with a built-in **✨ MiniMax H3 Image Fine-tune (5e-5)** preset: the whole model trains, not a LoRA, with Fizgig's training adapter on as for a LoRA and Likeness optimization on its usual schedule. The result is a full H3 Image file (~21 GB) that Studio, LoRA Royale and ComfyUI load as the base. It trains on a 24 GB card too: a smaller card works on the model a slice at a time, so each rotation takes more epochs. In our test on a person dataset, 5e-5 had the face by epoch 4 and a strong, clean likeness by epoch 30. A fine-tune of one person leans the whole model toward them, so use the Fine-tune card's regularisation set (pictures of the broad class) to hold that back.

**Studio works without a LoRA.** Leave the primary box empty and press Start: Studio renders the base model on its own, for any model, in one pane, with every setting, Save, Save every render and the Studio queue working as usual. The block sliders and Save repaired LoRA grey out with a line saying they need a LoRA. A donor on its own still compares, the base model against the base with the donor. **Unload primary**, beside the primary's Choose…, takes the LoRA out without reloading the model, as Unload donor does.

**Huge MiniMax H3 Image renders in Studio.** Width and Height go up to 12288 on H3 Image, so it renders at the sizes it was made for (4 to 20 MP, such as the model card's 4736 × 3552) and as long panoramas or tall strips, including 7680 × 2176 for a 32:9 ultrawide (crop 8 px top and bottom for 7680 × 2160). A render too big for your card says so.

**Studio says render, not preview.** Studio's own pictures, clips and songs are what you make there, so its words now say render and output: Rendering…, Render with:, the Output card. The Samples tab's training previews keep the word.

**The mask editor keeps three brush settings.** In Image Prep's Edit masks, buttons **1 2 3** (or the number keys) recall a tool with its Size, Softness and Strength; they start as Brush, Eraser and SAM click, and **Keep these settings…** stores what you're using into one of them, kept in this browser (#381, thanks @spaghettified26).

**To update:** run the updater as usual.
