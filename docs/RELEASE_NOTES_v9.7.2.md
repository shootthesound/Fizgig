# Fizgig v9.7.2: a regularisation set for LoRAs, AMD fixes, and Gizmo prepares a whole folder

**Train a LoRA with a regularisation dataset.** The Subjects card has a new fold at its end: point it at a folder of captioned pictures of the class you're training (other people, other cars, other rooms) and the LoRA learns your subject against them, so it bleeds less into everything else. In our testing the Subjects card's own modes, DOP and Fizgig LoRA Isolation, do better for most people, so start there; a regularisation set is for the cases where that approach is what you need. **LR ×** sets how hard those pictures pull (1.0 by default), and **Use each epoch** limits how many of them an epoch trains on, a different handful each time. Captions are required. While it's on, DOP, Fizgig LoRA Isolation and extra subjects are set aside, and the run keeps to one GPU. For Standard LoRA and LoKR on every still-picture model (#332, thanks @axymeus).

**Gizmo prepares a whole folder of videos.** On the video tab, **Prepare a folder…** (next to Open video) makes every video in a folder ready to train and writes them into a new folder, so your originals stay as they are. Each video gets 24 fps at real speed, its own size cut to a multiple of 32, its sound at 32 kHz stereo, and a length H3 takes, from its start. **Longest clip** sets how long a clip can be, with the smallest card that trains each length beside it: on a 16 GB card, pick 56 frames. A longer video is cut into several clips back to back, or just its start. Caption files come along with each clip, and a video that's already right is copied as it is. Thanks WiseDuck for the idea.

**Clips with any audio sample rate train as they are.** 48 kHz, 44.1 kHz and mono clips no longer need converting: Fizgig turns the sound into H3's 32 kHz stereo when it caches, and the files are never changed (thanks WiseDuck).

**Purge old runs from Runs & outputs.** Clear out every run that ended more than a number of days ago in one go. You see the list first and confirm twice. Tick **Delete their LoRAs and saved files too** to remove what they saved as well (the finished LoRA included); unticked, only the list lines go. Paused runs stay unless you include them (#345, thanks @spaghettified26).

**AMD:**
- **Krea 2 renders above 512 px came out wrong, and LoKR training could go to NaN.** ROCm's fast attention kernel gets the usual padding mask wrong. Krea 2 now needs no mask on AMD, and every model's masked attention gets the mask in the form ROCm handles correctly. Krea 2 training also uses 0.3 GB less (#341, thanks @scryptio for the bisect; #343, thanks @daxter93 and @xGnome).
- **Out of memory on an RX 9070 XT above 512 px** came from the same mask, which sent attention to PyTorch's slowest kernel (about 8 GB a call at 1024 px). Fixed by the change above (#343).
- **A PC with built-in Radeon graphics ran on those instead of the graphics card** when the ROCm build ignored the launcher's card choice, which crashed MOSS-Image caching with "invalid kernel file". Fizgig now makes the chosen card the one it uses, and says so in the log (#342, thanks @johnstephendonahue-ux).
- **RX 9070 / 9070 XT on Linux: optional SageAttention.** The Linux AMD installer builds SageAttention-RDNA4 for these cards, and **Preferences > Inference performance > Force SageAttention for Krea 2 image generation** uses it for Krea 2 pictures. Off by default, and only shown when it's installed. We couldn't test the kernel itself on a 9070, so tell us how it goes (#344, thanks @xGnome, and @daxter93 for the SageAttention-RDNA4 testing).
- **The Linux AMD installer works again.** It stopped at the PyTorch step (three missing helpers, found by @xGnome), read some cards' chip as the wrong one, and stopped at the face models before writing the launcher's settings.
- **ROCm installers:** the experimental and nightly package indexes can be overridden, and the Linux stable index is fixed (#340, thanks @scryptio).

**Helpers:**
- **A helper GPU renders MOSS-Image's SFT previews.** Previews on a separate preview model (MOSS-Image's SFT, Klein's Distilled, Qwen's Turbo) stayed on the training card while the helper sat idle; the helper now loads that model and renders them (#339, thanks @raymondgp).

**Fixes:**
- **ACE-Step: caching long songs no longer runs out of memory.** A 3-4 minute song went through the audio VAE in one go, which needed over 5 GB a layer and failed on 16 and 24 GB cards. Songs now encode in overlapping pieces: a 4-minute song peaks at about 3 GB, with the same result (#347, thanks @Aleksandr-AI90).
- **Krea 2 LoKR on a 16 GB card no longer slows to a crawl.** Auto now counts LoKR, the optimizer's memory and EMA, and keeps 4-bit with a warning saying what to turn off, instead of moving to a swapped int8 plan that ran about 6x slower (#337, thanks @8-lab1627).
- **Gizmo says why it didn't open.** If Fizgig's Python has no Tk, the Open Gizmo button says which package to install, and so do the Linux installers and the updater. On Linux, Gizmo uses your desktop's own file dialog when zenity or kdialog is installed (thanks WiseDuck).
- **MOSS-Image:** the Training tab's status line now gives the current Clean-end share Slide defaults (11% until epoch 7, then 40%).

**Also in this release:**
- **Preferences > App window: Disable GPU for the app window** on Windows and Linux, so the window holds none of the VRAM a run needs. It replaces 9.7.1's Draw the window without the graphics card; tick it again if you used that (#325, thanks @scryptio, and @daxter93 for testing).

**To update:** run the updater as usual. On Linux, if it says Tk is missing, install the package it names to use Gizmo.
