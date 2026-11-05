# Fizgig v9.0.6: Identity mode for Extract, and batches for fine-tunes and the loss watch

**Identity mode in Extract.** A character's likeness lives mostly in a few blocks, so a LoRA extracted at an even rank can lose resemblance that the full fine-tune had (#301, thanks @pomoe136). Tick **Identity mode** on Extract, under Fine-tune to LoRA or From a LoRA, and the model's identity blocks keep 4× the rank you choose while every other block keeps that rank: rank 128 keeps 512 where the likeness is. The file grows far less than raising the rank everywhere. It works for MOSS-Image (blocks 24–35) and Qwen Image 2.1 (blocks 10–14), and the saved LoRA has `_id` in its name.

MOSS-Image's identity blocks come from profiling a character LoRA, and a run training only those blocks gave superb likeness. With them, MOSS also gets what Qwen already had from its map: **Fast Identity Mode** on the Training tab, and Studio's identity and look colours with the Identity only, Look only and Identity ×0.5 presets.

**Batch size for full fine-tunes.** This builds on 9.0.5, which added Batch size (on desktop cards, most useful for smaller models such as Anima and SDXL). A fine-tune can now train several pictures a step too. Fizgig leaves room for the extra pictures before it sizes the fine-tune's windows, and the fine-tune card's plan line shows the result for your card. Regularisation pictures keep their own learning-rate multiplier inside a batch, and Move to… gives a fine-tune the largest batch that keeps its window plan. DOP and Start at a smaller size still train one picture a step.

**The per-image loss watch works at any Batch size.** Detect problem images, Per-image adaptive LR, the look-outlier warm-up and Auto-recaption no longer turn off when Batch size is above 1. Each picture in a batch is weighed, excluded and recorded on its own, as its own step would be. Thanks to @Murf-Code, whose Fizgig Advanced did this first.

**Also new**
- The Profiler's subject photos are chosen in the same picker as LoRA Royale's and the Samples gallery's: thumbnails of a folder, and **Choose folder…** for any other.
- The CLI guide notes that Batch size's speed-ups were measured at 0.25 MP. At 1 MP the gain is smaller unless blocks are swapping.

**Updating:** update Fizgig on your computers, and restart your pods.
