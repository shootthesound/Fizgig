# Fizgig v9.0.1: pods managed from the Pods tab, Start that downloads the model, and an auto-recaption fix

**Auto-recaption no longer stalls mid-run.** Since v8.9.95, recaptioning a stuck image during a run could take hours on a 16 GB card: the captioner saw the training model on the card and streamed every layer from system RAM for each word. It now goes on the card whole whenever it fits, as it did in v8.9.94 (thanks @Klow666, #275).

**Start downloads a model you don't have yet.** Press Start on a model whose files aren't downloaded and Fizgig runs **Download models for me** for it, then starts the run by itself once the files are in. The Training tab shows the download's progress with a Cancel button. Klein and Flux 3 files need your Hugging Face token: give it on the Preferences tab first.

**Krea 2 loads ComfyUI's int8 builds.** The Krea 2 DiT slots take ComfyUI's int8, int8 ConvRot and other fp8 checkpoints, so a Krea 2 file you already have for ComfyUI works in Fizgig (thanks @crazyscale-dotcom, #273).

**The Pods tab manages your RunPod pods.**
- **Delete pod** deletes a pod on RunPod, after a confirm in its row that says what goes with it (the pod and its volume disk) and, if it's training, to use Bring back first.
- After **Start** or **Stop**, the row shows Starting... or Stopping... while RunPod catches up, instead of flicking back.
- A stopped pod links to your RunPod account for what it costs while stopped.
- Your RunPod pods join the Helpers list by themselves once their Fizgig answers, with no job ticked. Remove one and it stays removed until you add it by hand.

**On a pod, the sidebar shows the file manager login** above its button: user `admin` and the password, on Show. The Disk meter shows the mount point instead of the whole path, and its tooltip says a network volume reports the whole shared storage, not your volume's size.

**Fixes**
- Training folder tiles on the Start tab (and in From this computer) show pictures for a folder that keeps all of them in sub-folders.
- The Likeness picker shows each training picture once: with Detail crops on, it listed every picture twice.
- The Style score says where its scores are: the style % at the bottom left of each preview.

**Docs:** the MOSS-Image guide recommends **LoraLoaderBypassModelOnly** for the int8 SFT file in ComfyUI, and the CLI guide covers every model and the newer training flags.

**Updating:** update Fizgig on your computers, and restart your pods.
