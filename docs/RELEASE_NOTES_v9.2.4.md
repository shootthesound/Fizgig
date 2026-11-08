# Fizgig v9.2.4: several edit instructions in Edit LoRA previews, and faster MOSS-Image previews on 16 GB cards

**Edit LoRA previews show several edits** (#311). The **Edit instruction** box on the Edit pairs card now takes one instruction per line, like the Samples prompts, and every line is previewed on your test photo each round, so a LoRA trained on a family of edits can show all of them. **Write captions** saves only the first line as the photos' caption, and has a tooltip saying what it does. Thanks to @CognitiveDiffusion.

**Faster, smaller previews for MOSS-Image on cards under 20 GB.** Since 9.2.2, MOSS-Image previews render at full size on every card, because shrunken ones looked broken. If you'd rather have speed, the Samples tab now has a **Faster, smaller previews (768 px)** tick, shown only on cards under 20 GB and off by default. Its note says the size the previews will actually render at, and the gallery marks previews made smaller. They're softer than full size, so use them to watch progress, not to judge quality.

**Training speed shows training only.** The run card's Speed now leaves out everything between epochs (saves, checks and previews), so it's the speed of the training steps themselves. Before, only previews were left out.

**Updating:** run the updater as usual, and restart your pods.
