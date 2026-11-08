# Fizgig v9.2.5: MOSS-Image Turbo Train, and new Clean-end share defaults

**New preset: ✨ MOSS-Image Turbo Train (Experimental).** It's the Fast preset (rank 16, 0.25 MP, Muon) with Adaptive LR on between 2e-4 and 4e-4, and per-image tracking with per-image adaptive LR on, so the learning rate climbs while the LoRA is improving and each picture gets its own pace. It can work very well on a high-quality dataset. On any dataset, lesser quality included, it's also a quicker way to preview what a LoRA could become before committing to a longer run.

**MOSS-Image's Clean-end share is reversed: `30:7:60:8:3`.** The LoRA, Style and Slider presets now spend 30% of steps on nearly clean images until epoch 7, then 60% from epoch 8, with every 3rd epoch after that at 30% and half the learning rate. With Muon, now every preset's optimizer, this order gave the best results. The Fine-tune preset keeps `11:4:30:8:2`. Load a ✨ preset to pick it up: saved settings keep what they had.

**MOSS-Image also**
- **A note under the model picker** says to start from a ✨ preset, to train with the new Muon optimiser and the new Clean-end share defaults.
- **Faster, smaller previews is on by default on cards under 20 GB** until you choose. Untick it for full-size previews; your choice is kept.

**Per-image loss has a Refresh button** beside its sort menus, and changing either menu loads the latest results too, so there's no need to close and reopen it while a run trains.

**About the pace of updates.** I'm pushing small updates often at the moment: this is a period of optimising Fizgig from your feedback, what I see in real runs, and tests. You don't need to chase them on pods: a pod you use from your own Fizgig (opened in the app window, or in your Pods list) switches itself to the same version as your computer when it's free, and keeps it for later starts. So update your own Fizgig when it suits you, and your pods follow, without restarting them every time. You get the improvements as they land, and a setup that stays in step.

**Updating:** run the updater as usual.
