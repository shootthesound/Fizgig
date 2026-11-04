# Fizgig 9: Refinements, and Detail crops for styles

Fizgig 8 rebuilt Fizgig as a web app. Fizgig 9 brings together the 63 releases of refinements since then: an app that explains itself, runs that survive things going wrong, and pods that work like your own computer. Some of Fizgig's tools were made with character LoRAs in mind (likeness scores, face crops, ArcFace); new in 9, Detail crops and a Style score give a style LoRA the same care.

## New in 9

**Detail crops, for style LoRAs.** A new Image Prep option: each picture also trains as one square crop at its own scale, so brushwork, canvas grain and texture train pixel for pixel instead of being shrunk away with the whole picture. The crop is at a random spot, moved off flat areas such as a dark background, and the same spot at every size. Image Prep makes the crops at every size up to the one you pick, and each run makes any size it needs at Start, Start at a smaller size included.
- **On the Captions tab** the crops are listed after your pictures. Each picture's crop is captioned once, on its largest size, and that caption is copied to the crop at every other size, whether the AI wrote it or you did.
- **Train on the crops only** moves the full pictures into a `detail_masters` folder that never trains, but still feeds the crops.
- The crops go with the folder when you move a run, download its zip or send it to a pod.

**A Style score.** The Samples gallery has a **🎨 Style score** button beside Likeness. Switched on, it scores each preview against your training pictures' look as a whole: texture, palette, line and medium more than who is in the picture. Each preview's score (0 to 100%) is how close it is to its 8 nearest training pictures. It runs on the CPU, so training keeps the GPU, and the first time it downloads DINOv2 (about 350 MB). Likeness says nothing about a style LoRA; this gives it a number to watch epoch by epoch (thanks @Murf-Code).

**Caption tasks for every kind of LoRA.** Every caption length comes in a person or character version and an object or product version. **Style** is there for a style, and a **concept** task for a pose, an action or an effect names it the same way in every caption, in the words you give in **The concept is** box. Auto-recaption offers them all.

**Steadier Adaptive LR.** Adaptive LR judges a plateau on the average loss of the last 3 epochs instead of one epoch's, so one jittery epoch no longer halves the learning rate while the run is still improving. The log shows each epoch's loss and the 3-epoch average, and a resumed run carries the average on (thanks @Murf-Code).

**Arrow keys in the preview viewer.** With a preview enlarged, ← and → show the same prompt an epoch earlier or later, and ↑ and ↓ the other prompts at the same epoch.

**Image Prep shows only what each choice uses.** The face rows show only for the face close-up modes, and Detail crops hides the originals choice, which doesn't apply to it.

**Mask painter.** The brush outline shows on white, black and any picture, and a right-click erases with the Brush and paints with the Eraser (thanks @mart-hill, #271).

## Since 8.0

If you're coming from 8.0, this is what changed in between. Each release's notes have the details.

**Three new models, and the others made right.** **Flux 3 Image Klein** (8.1), **MOSS-Image** (8.5) and **Z-Image Base** (8.9.95) joined, each with LoRAs, LoKR, sliders, fine-tunes and the workbench. MOSS-Image took the most work: previews that match ComfyUI, all 42 blocks trained, and the **Clean-end share** schedule, which brought likeness in far sooner and holds the composition steady from epoch to epoch. It trains on 16 GB cards and on computers with 16 GB of RAM. Krea 2 now loads on little RAM, and Auto stopped picking a slow setup on 16 GB cards.

**New ways to train.**
- **Masked training** (8.9.6): train on the part of each picture that matters, with masks made in one click, from words with SAM 3.1, or painted by hand in the mask editor.
- **Fizgig LoRA Isolation** (8.4): two identities in one LoRA, with everyone else still looking like themselves.
- **Edit LoRAs with extra pictures** (8.1.2), **prompt sliders that leave chosen prompts alone** (8.9.93), **train on sub-folders** (8.9.92), **caption dropout** and **Start at a smaller size** (8.9.98).

**An app that explains itself.** The Training tab shows only what a run uses, in grouped boxes with their explanations a click away. **Start** warns about the usual mistakes (pictures with no caption, a missing trigger word, a learning rate far outside the model's presets) and lets you start anyway. The wording covers styles, objects and concepts as well as people. There's a **Run summary** under the run card, a **table view** of your captions, meters in the sidebar for disk, background jobs and what's loaded on the GPU, and **eight themes**, three of them for eyes.

**Runs that survive things going wrong.**
- A run that runs out of GPU memory ends with **Continue** from its newest save.
- A failed save retries instead of ending the run, and Resume rebuilds a cache you deleted.
- A folder refilled with new photos under the same names trains on the new photos.
- Every run's log is saved to a file, and Start can't start the same run twice.

**Changing your mind mid-run.** Change previews while a run trains, steps included. Pause, change the learning rate, block swap, save schedule or picture size, and **Resume** takes them; each is marked **Applies on Resume**. **Continue** any run from Runs & outputs, so you can put one project down, train another and come back.

**Your other computers and the cloud, as one Fizgig.**
- **Helpers** (8.9): a second GPU, in this computer, another or a pod, trains a share of each epoch or renders your previews so training never pauses for them.
- **Move a run** to a laptop, another PC or a pod and bring it back. Its checkpoints and previews come home as they're saved, and the pod can delete itself once everything's back.
- A pod runs the same Fizgig as your computer, logs you in by itself, and lists only cards you can rent right now. On cards of 40 GB and up, training is 13–27% faster.

## Coming next

**FIBO 2.0**, Bria's new 8B open-weights model for making and editing pictures, is on its way to Fizgig: @spaghettified26 has opened a pull request that trains LoRAs on FIBO 2.0 Dev (#250).

## Thanks

To everyone who reported, asked, tested and contributed since 8.0, and especially to @Murf-Code for MOSS-Image, faces for anime and furry characters, the Style score and more.

**Updating:** update Fizgig on your computers, and restart your pods.
