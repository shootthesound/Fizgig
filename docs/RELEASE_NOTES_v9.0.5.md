# Fizgig v9.0.5: Batch size

**Train several pictures a step.** Batch size, under Other Options on the Training tab, now goes up to 8 on every model except ACE-Step (thanks @Murf-Code and @tsk5cb-cmyk, #221). A step's loss is the average of each picture's own, so a batch learns as Gradient Accumulation does, faster wherever the card has room for it. At Batch size 1 nothing changes: runs train exactly as before.

It pays most with block swap, because each step's swap then serves every picture in it. Qwen Image 2.1 with 8 blocks swapped went from 5.0 s a picture to 1.9 s at batch 3, and MOSS-Image ran 1.8× faster. With no swap, the smaller models gain most: Anima ran 2.6× and SDXL 2.3× faster on a 4090. The large models gain little when they already fit without swapping. Under the Batch size box, Fizgig shows what the batch needs on your card (Auto's precision and block swap at batch 1 and at the batch) and whether it's likely to be quicker.

A batch makes fewer, averaged updates each epoch. At batch 4, with the same epochs and learning rate, a run makes a quarter of the updates, so it may need more epochs or a higher learning rate to get as far.

On MiniMax H3 the batch is made of pictures; a clip or a voice still trains on a step of its own. Some features read one picture a step, so they stay at batch 1: edits, sliders, fine-tunes, DOP, Fizgig LoRA Isolation, the per-image loss watch, Start at a smaller size, keeping the model's own picture outside the mask, regularisation images, helper shares and H3's reference distillation. The Training tab greys them out while Batch size is above 1.

**Move a run to a bigger card and batch it there.** Move to… has a new tick, **Train several pictures a step there**. The pod works out the largest batch its card fits at the run's precision without swapping more blocks. Bring the run back and it carries on at your own Batch size, with the learning rate schedule following on from its epochs.

**Fixes**
- Flux 3 Image Klein: **Download models for me** works again. The model's files were renamed on Hugging Face on 17 Oct, and the old names no longer downloaded.
- Klein 9B's VAE is now saved as `flux2-vae.safetensors`, the same file Flux 3 and Fibo 2.0 use. It used to be saved as `ae.safetensors`, the name of Z-Image's different VAE, so downloading Klein and Z-Image one after the other gave the second one the first one's VAE. Fizgig now spots a VAE row pointing at the other model's file when it starts, and moves it to the right one. If the right file isn't on your computer, the row is cleared, and **Download models for me** (or Start) fetches it.

**Updating:** update Fizgig on your computers, and restart your pods.
