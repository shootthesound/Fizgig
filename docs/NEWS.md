# Fizgig news

## 8 Nov 2026: Faster MOSS-Image training and previews, faster training on every model, and a faster Studio

**Training is much faster, uses far less memory, and likeness is unchanged.** On an RTX 5090, a MOSS-Image LoRA at batch 1 (94 pictures, Muon, 70 epochs) takes about 20 minutes instead of 49. At batch 3, MOSS-Image now matches OneTrainer's speed on the same card in less memory (1.07 s a step in 6.9 GB, against 1.07 s in 7.4 GB). Thanks to @tsk5cb-cmyk, whose logs in #221 showed where the time was going.

**Quick MOSS-Image previews that still look right.** Previews now render on the Base with the Turbo LoRA at 0.6, 8 steps, CFG 2 and the SFT look LoRA: much quicker than full-quality previews, and a good read of how the LoRA looks. Both LoRAs download with the models. Untick Use the Turbo LoRA for full-quality previews.

**A faster Studio.** Studio, LoRA Royale and the Profiler follow the Samples tab, so on MOSS-Image they render with the same quick recipe. On MiniMax H3, Show early's mid-render look takes a third of a second instead of a second and a half.

**What makes training faster.** PyTorch 2.15 keeps far less in memory with gradient checkpointing on (Krea 2 22.9 → 14.9 GB). Muon's step handles a LoRA's matrices together: 12× faster, same results. Batches with captions of different lengths no longer slow attention down. LoKR runs 4–6× faster on every model, including as a Context LoRA (thanks @Nessa2026691, #307).

**Also:** onnxruntime's and Hugging Face's telemetry is off, and Adaptive LR learns a ceiling.

**Your NVIDIA driver must support CUDA 13** (version 580 or newer). The update downloads PyTorch 2.15 (about 3 GB); if Fizgig won't start afterwards, run the updater again and press R to rebuild its Python environment. Fizgig now needs Python 3.11 or newer: the release notes say what to do if yours is on 3.10. AMD installs keep their own PyTorch. Restart your pods to pick up 9.2. [Fizgig v9.2.1 release notes](https://github.com/shootthesound/Fizgig/releases/tag/v9.2.1)

## 6 Nov 2026: Muon for every picture model

**Improved likeness and training performance across most of the models, thanks to the Muon optimizer.** Every picture model's built-in presets now train on Muon: Klein 9B, Flux 3 Image Klein, Fibo 2.0, Krea 2, Qwen Image 2.1, MOSS-Image, Z-Image Turbo and Base, SDXL, Anima and Vimoo 1.0 Lite.

**Why it helps.** Muon steps each weight matrix as a whole and evens out the update across all its directions, so fine detail (skin texture, hair, fabric) learns as fast as the broad shapes instead of waiting behind them.

**Nothing else to change.** Fizgig scales Muon's step to match AdamW's, so the presets keep their learning rates. It also uses less memory than AdamW. Load a preset again to get it: the Training form and your saved presets keep the optimizer they had. AdamW8bit and AdamW are still in the Optimizer Type row. Full fine-tunes keep Adafactor, and MiniMax H3 and ACE-Step keep their own optimizers. Thanks to @Murf-Code, who brought Muon to Fizgig.

Update Fizgig on your computers, and restart your pods. [Fizgig v9.0.8 release notes](https://github.com/shootthesound/Fizgig/releases/tag/v9.0.8)

## 6 Nov 2026: Vimoo 1.0 Lite

**Fizgig trains Vimoo-AI's 9.5B video model**, released under Apache 2.0: one checkpoint for text-to-video and image-to-video at 720p and 24 fps. In our tests its quality and performance sit closer to Wan 2.2 5B than to the newer video models; it's a young model and may improve with later releases.

**Why Lite, and not the full Vimoo 1.0.** If your hardware can run the full Vimoo 1.0, there are probably better video models you can run instead. The full model is two ≈11.7B checkpoints, one per task, working on four times as many tokens per clip. Lite's reason to exist is that it's light, and it trains on a 16 GB card.

**What you get:** LoRA and LoKR from photos and clips, slider LoRAs, full fine-tunes, masked training, and an image-to-video share that trains part of each clip from its own first frame. Training previews render as clips in 12 steps with Vimoo's distilled LoRA; Studio, the Profiler and LoRA Royale work with it, and Studio can start a clip from a picture. **Download models for me** on Preferences fetches the files (about 40 GB), and LoRAs load in ComfyUI with Vimoo's own nodes.

Update Fizgig on your computers, and restart your pods. [Vimoo guide](https://github.com/shootthesound/Fizgig/blob/master/docs/VIMOO.md) · [Fizgig v9.0.7 release notes](https://github.com/shootthesound/Fizgig/releases/tag/v9.0.7)

## 25 Oct 2026: Masks from words with SAM 3.1

**Make masks of anything by naming it.** On the Image Prep tab's masks card, choose **SAM 3.1: from words** and type what the masks show ("person", "hair", "the dog", "red dress"), and optionally what to leave out ("glasses"). Meta's Segment Anything model finds it, with much sharper edges than the person finder, around hair especially.

Download it once from the card (1.75 GB), press **Preview on 2 random pictures** to check, then **Make masks**. It runs on the GPU, so make masks when nothing is training. Update Fizgig on your computers, and restart your pods. [Masked training guide](https://github.com/shootthesound/Fizgig/blob/master/docs/MASKED_TRAINING.md) · [Fizgig v8.9.9 release notes](https://github.com/shootthesound/Fizgig/releases/tag/v8.9.9)

## 25 Oct 2026: Masked training

**Train on the part of each picture that matters: the person, not the room around them.** A black-and-white mask beside a picture says which part counts. White is trained fully, grey partly, and black only as much as you choose. The picture itself still trains whole, so the LoRA sees your subject in their scene; it just stops copying the scene.

**In our tests of character LoRAs, masks significantly raised both the best likeness a run reached, by eye and in likeness scores, and how quickly it got there, on every recent model. It seems especially effective on MOSS-Image with its default Clean-end share, Slide.** It works for Klein, Krea 2, Qwen Image 2.1, Flux 3, MOSS-Image, Z-Image, SDXL and Anima.

**Make masks in one click** on the Image Prep tab: the person, the person without their clothes, just the face and hair, the body and clothes without the face, the body's skin only, just the clothes, everything but the faces, or just the background (for a place or scene LoRA). With several people in a picture, it can keep just your subject. Or paint your own, or bring masks from another trainer that uses the same naming.

**Prior preservation outside the mask** came in 8.9.7: with **Outside the mask, keep the model's own picture (prior preservation)** on, outside the mask the LoRA learns to draw what the model itself would there, so backgrounds and the model's look stay the model's own. [Fizgig v8.9.7 release notes](https://github.com/shootthesound/Fizgig/releases/tag/v8.9.7)

Tick **Train on the parts your masks show** on the Training tab. Update Fizgig on your computers, and restart your pods. [Masked training guide](https://github.com/shootthesound/Fizgig/blob/master/docs/MASKED_TRAINING.md) · [Fizgig v8.9.6 release notes](https://github.com/shootthesound/Fizgig/releases/tag/v8.9.6)

## 24 Oct 2026: Faster training on 40 GB+ cards

**Training is faster on cards with 40 GB or more**, which is most pods: A100, A40, A6000, L40S, H100, RTX PRO 6000. Gradient checkpointing saves memory by working parts of each step out twice; a big card doesn't need the saving, so Auto now turns it off when the run fits with room to spare. That's 13–27 % faster per step, measured on Krea 2, Qwen Image 2.1, MOSS-Image, Z-Image and MiniMax H3 (stills and clips).

**Where it doesn't all fit**, at bigger sizes, the last blocks go without it instead: 2–21 % faster.

**Previews keep at least 4 GB free** either way. The figures behind it were measured on A100 80 GB, A40 48 GB (and one capped to 40 GB) and RTX PRO 6000 cards, from 0.25 to 2 MP. Cards of 32 GB and under, rank above 32, LoKR, fine-tunes and sliders keep checkpointing on, as before.

Restart your pods to get it. [Fizgig v8.9.5 release notes](https://github.com/shootthesound/Fizgig/releases/tag/v8.9.5)

## 22 Oct 2026: Your other GPUs, at home or in the cloud, speeding up your run

**▶ [Watch: Your other GPUs speeding up your run](https://www.youtube.com/watch?v=7Ao0gE2b2qw)** (2 minutes)

**A helper is another computer running Fizgig:** a second PC or laptop on your network, a second GPU in this computer, a RunPod pod or a Vast.ai instance. Add one on the new **Helpers** tab: the computers on your network and your running pods are listed for you.

**Previews that never stop training.** Switch **Previews** on, on a helper's tile, and it renders your previews while your GPU keeps training. Click a preview to see which computer rendered it.

**Move a run.** Need your computer back? **Move to…** on the run card pauses the run here and carries it on on the helper. **Bring it back** when you want it; if the other computer goes away, **Carry on here** takes the run back from your last save.

**Train on two GPUs** (experimental). Switch **Training share** on and the helper trains a share of each epoch alongside your GPU. Each GPU trains its own copy of the LoRA on its part of the images, both starting from the same point; at every combine their changes are added into one LoRA, and both carry on from that. So you still get one LoRA that has learned from every image: on a Krea 2 test, combining every 3 epochs, its likeness matched a one-GPU run from the 9th epoch on. Every 3 epochs is the default; **on a fast network** (two GPUs in one computer, or your home network) **combine every epoch** for the closest result to one GPU, at very little time cost. The run waits for the helper before its first epoch so both start together, and the helper's tile says round by round whether the share is saving time. It pays most on long runs, where the time saved adds up.

**A second GPU in this computer:** **Use it as a helper** beside **This computer's other GPU** starts a second Fizgig on that card.

**Your home GPU helping a pod:** **Help another Fizgig with this GPU** renders a pod's previews or trains a share of its run. Tick **Bring its checkpoints home as they're saved**, and the pod can delete itself once they're all home and checked.

**Renting a GPU:** the **Pods** tab lists the RunPod cards free right now, cheapest per run first, with one card or two. On RunPod's page, add a 300 GB **Volume disk** before you deploy ([how](https://github.com/shootthesound/Fizgig/blob/master/docker/README.md#give-it-a-disk-on-runpods-deploy-page)). Fizgig runs on [Vast.ai](VAST.md) too.

Both computers need this version: update Fizgig on your computers, and restart your pods. [Helpers guide](HELPERS.md) · [Training share](TRAINING_SHARE.md)

## 20 Oct 2026: MOSS-Image likeness sooner, and Base previews with the SFT look

**▶ [Watch: Training MOSS-Image LoRAs in Fizgig 8.8.1](https://www.youtube.com/watch?v=UCoCtM8yrBk)** (2 minutes)

**A new Clean-end share default for MOSS-Image: Slide.** It puts 60% of training steps on nearly clean images, where faces and fine detail are learned, until epoch 7, eases down to 30% by epoch 15, and then sends every 2nd epoch back to 60% so likeness doesn't fade. Likeness comes in early, then composition. On the dataset that had given us the most trouble, it was the best of eight settings we tried: around 70% ArcFace likeness from epoch 20 to epoch 60, a very high score (different real photos of the same person often score lower). It's tuned on datasets of up to about 55 images; with many more, a steady 40% with **Single** may suit better. Every MOSS-Image preset uses Slide. A saved form keeps its old value: pick a preset, or pick **Slide** on the Clean-end share control.

**Give Base previews the SFT look** (Samples tab): with **Use the SFT model for samples** off, Fizgig's SFT look LoRA gives your Base previews much of MOSS-Image SFT's look, with no model swap, so previews are much quicker. It's not as good as SFT itself, but a good guide. It downloads itself the first time you tick it (about 2.1 GB). **We still recommend not using the Turbo LoRA for previews.**

**Each enlarged preview says how it was made:** SFT, the Base with the SFT look LoRA, or the Base alone, and whether the Turbo LoRA was on.

**Update ComfyUI.** Many people tell us their Fizgig LoRAs work better since ComfyUI's update this week.

## 20 Oct 2026: MOSS-Image Turbo previews: fast, but don't judge likeness on them

**The MOSS-Image Turbo LoRAs aren't mature yet: they can reduce likeness a lot in previews.** Several people judged a run's likeness on Turbo previews and thought it wasn't learning. Use the Turbo LoRA to check quickly that a run is on track, and judge likeness on full-quality previews: set **Turbo LoRA strength** to 0 on the Samples tab, and Steps and CFG go back with it. With **Change the previews of the run going now** ticked, that applies from the run's next preview. The gallery's amber Turbo line now says this too.

**New for every model: How often to preview and How often to save.** Each is one control: **Every** N epochs, and, switched on, **then from epoch** Y, **every** Z epochs. Switching it on fills in every 5, then from epoch 10 every 1: few previews or checkpoints early, every epoch once the run gets close. The last epoch is always saved, and a fine-tune keeps saving at whole rotations.

## 20 Oct 2026: Kromoss out as a preview model, the Turbo LoRA stays

**Kromoss is no longer offered as the MOSS-Image preview model.** It's too different from MOSS-Image Base for reliable previews of a LoRA trained on Base: side-by-side renders of the same LoRA, prompt and seed on SFT, Kromoss and Base showed it. Previews render on **MOSS-Image SFT** again, and the Preferences row is back to **MOSS-Image SFT DiT (previews)**.

**If you set Kromoss on a MOSS-Image row** (the preview model in place of SFT, or the Base DiT in place of Base), go to Preferences → MOSS-Image and press **Download models for me**: it points each row back to SFT and MOSS-Image Base, downloading whichever you don't have. Your Kromoss files stay on disk.

**Train on MOSS-Image Base, not on Kromoss:** in our test, a LoRA trained on Kromoss began to overbake after about 750 steps (epoch 12 on a 62-image dataset), where the same training on Base held up.

**The Turbo LoRA stays.** lodestones' 8-step Turbo LoRA, distilled from MOSS-Image Base, gives fast previews of your Base: **Download** on the Turbo LoRA row in Preferences, untick **Use the SFT model for samples** on the Samples tab, and set **Turbo LoRA strength** to 1. Steps move to 12 (much better than its card's 8) and CFG to 1. You can turn it on mid-run too: it joins from the next preview. Previews on SFT never take it.

## 19 Oct 2026: MOSS-Image training, fixed

**MOSS-Image LoRAs now learn likeness much sooner, and the picture holds its composition from epoch to epoch.** There's nothing to set: update Fizgig and it's the default.

**What was wrong.** A diffusion model trains at every noise level, from nearly clean pictures to pure noise. The clean end is where faces and fine detail are learned; the noisy end is layout and composition. MOSS-Image's own training schedule puts only about 11% of steps at the clean end. That's why MOSS LoRAs were slow to pick up likeness, and why the picture jumped about between epochs.

**What Fizgig does now.** A new Training-tab setting, **Clean-end share**, puts 30% of training steps at the clean end, close to where Krea 2 and Qwen Image sit, the models that pick up likeness fastest in my own training. It changes training only; how MOSS-Image generates pictures is unchanged.

ArcFace likeness against 3 reference photos, same dataset:

| Epoch | 3 | 6 | 9 | 12 | 15 | 20 |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| Before (the model's own schedule) | 40% | 44% | 52% | 58% | 63% | 60% |
| Now (Clean-end 30%) | 42% | 59% | 63% | 66% | 69% | 69% |

The old settings peaked at 67% after 30 epochs. On the Standard preset (rank 32, 1 MP) a second dataset reached 72% by epoch 32, above the 71% that the dataset's own photos score against the same reference photos.

**By eye the difference is bigger still: the pictures are far sharper and more accurate, with no body horror, and the composition holds steady instead of hopping between epochs.**

**Fine-tunes too.** A MOSS-Image fine-tune uses the same Clean-end share, and it fits a 16 GB card, the whole model at once: a full Base fine-tune peaked at 12.6 GB at 1 MP.

30 is still a figure to explore; share what you find in the [Reddit discussion](https://www.reddit.com/r/StableDiffusion/comments/1x9khtm/a_potential_fix_for_moss_training_issues/). Empty the box for the model's own schedule.

