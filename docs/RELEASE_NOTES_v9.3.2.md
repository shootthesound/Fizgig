# Fizgig v9.3.2: The trigger-word map on every still-picture model

**The Training maps' Trigger word map now works on Klein 9B, Flux 3 Image Klein, Fibo 2.0, Z-Image Turbo and Base, and Anima**, as well as MOSS-Image, Krea 2 and Qwen Image 2.1. It shows how much of each area's attention goes to your trigger word inside the model, epoch by epoch: for a person it should settle on the face and body. Before training, most models also put a made-up trigger word on any lettering in the picture, such as a logo or watermark, since they read it as text; watch it move onto your subject as the LoRA learns. SDXL doesn't have it: its map is too coarse at training sizes to show anything useful. [Training maps guide](https://github.com/shootthesound/Fizgig/blob/master/docs/TRAINING_MAPS.md)

**The Trigger word map is easier to read.** Some models spread a new word's attention almost evenly over the picture, which drew the whole picture in one colour. The map now shows where the attention is higher than in the rest of the picture.

**Docs:** the command-line guide covers the training maps flags, every optimizer and the fine-tune window flags; the guides for adding a model cover training maps and the trigger-word map.

**Updating:** run the updater as usual. Pods you use from your own Fizgig follow your computer's version when they're free.
