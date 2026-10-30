# Fizgig news

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

