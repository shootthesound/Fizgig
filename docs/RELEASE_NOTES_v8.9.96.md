# Fizgig v8.9.96: captions for products, objects, concepts, styles, people and characters, and Studio from any LoRA

**Caption tasks for objects and products.** Every caption length on the Captions tab now comes in two versions: one for a person or character (viewpoint, face, pose, clothing), and one for an object or product (its shape, colours, materials and any text on it, how it's placed, viewpoint and framing). The person versions are unchanged. Auto-recaption's **First try uses** and **Second try uses** offer both, so an object set can recaption with the object Training caption, then the object Exhaustive detail, the same way a person set does. For a style, **Style** is still the task to pick (thanks @bkrew, #244).

**A caption task for concepts.** For a pose, an action or an effect, pick **Training caption (concept: a pose, action or effect)** and say what it is in the new **The concept is** box, in a few plain words (*a backflip*, *a double exposure*). Every caption then names the concept in exactly those words, after the trigger word, and describes everything else in the picture, so the concept is named the same way throughout and the rest stays promptable. Auto-recaption uses the same words when it picks this task.

**Wording for every kind of LoRA.** Much of Fizgig's wording was written for person LoRAs. Hints across the tabs now say plainly what applies to a person, a style or an object:
- The face tools (Look Filter, likeness on the Samples tab, in Studio and in LoRA Royale) say they score faces, and what to do for a style or object LoRA instead.
- **Fast Identity Mode** says it's for a person or character LoRA, and to leave it off for a style or concept.
- **Masked training** and the **Subjects** card now cover any subject, not just people.
- The Samples tab's example prompts put the trigger word first, as captions do.

**Profiler: a LoRA's own prompt, and a report for every kind of LoRA.** The Profiler starts with the first Samples prompt of the run that made the LoRA. The bleed check gets the same prompt without the trigger word. So a style LoRA is profiled on the prompt you previewed it with, not a portrait. LoRAs with no run on record still start with the portrait. When the report can't measure likeness, it explains what it measured instead: either no face was found in the renders, the LoRA barely moves the faces, or no subject photos were given.

**Open a LoRA in Studio from where you find it.** Each LoRA on the Training tab and on Runs & outputs has a **Studio** button, and so does each preview whose epoch saved a LoRA. It opens that LoRA in Studio on the model its file was trained for, read from the file itself; press Start there to load it. The buttons are hidden while a run is training, so Studio never shares the GPU with it.

**ACE-Step's first preview prompt** is now a song description, not "A high quality photo".

**Fixes**

- **Start and Resume while the Profiler is rendering** started the run alongside it on the GPU. They now ask you to wait for the profile or stop it, as they already do while captions are being made.

**Updating:** update Fizgig on your computers, and restart your pods.
