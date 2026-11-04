# Fizgig v8.9.98: start at a smaller size, a run summary, and a tidier Training tab

**Start at a smaller size.** Training Parameters has a new **Start at a smaller size** choice: the first part of a run (20%, 30% or 50% of its epochs) trains at a smaller Target Megapixels, from 0.25 MP up, then blends into your full size over a tenth of the run. In a Z-Image Turbo test, 0.25 MP for the first 30% of a 1.0 MP run took about a fifth less training time, with likeness arriving a little later, so allow a few more epochs. It's for standard LoRAs on every picture model; edits, sliders, fine-tunes and MiniMax H3 train at one size as before. The smaller size gets its own cache, made with the run's other cache stages.

**A run summary.** Under the run card on the Training tab, a **Run summary** shows the run at a glance:
- **Time:** training time, total time, speed, epoch, the run's peak VRAM and time left.
- **Learning:** the learning rate the run is on now (it moves with Adaptive LR), the grad norm, the loss and, once the Samples gallery has scored an epoch, its likeness.
- **Setup:** model, dataset, epochs and size, optimizer, network and memory, from the run's own settings, so it stays right while you set up the next run. **Full spec** lists every setting it started with.

**A tidier Training tab.** Settings that belong together now sit side by side in one box each, whatever the window's width: Learning rate (with Adaptive LR on, its Min and Max take the Learning Rate box's place), Network, Length, Picture size and Context LoRA. Each box's explanations are behind its **?**, and other long explanations show two lines until you click them.

**Warnings before you start.** Besides what stops a run, **Start** now also lists the usual mistakes, with **Start anyway**: pictures with no caption, captions without the trigger word, and a learning rate well outside what the model's presets use. **Check** lists them too. Nothing is blocked.

**Caption dropout for every picture model.** **Other Options → Caption dropout** (Off, 0.05 or 0.10) trains that share of steps with no caption, so the LoRA leans less on the trigger word and the caption's exact words. MiniMax H3 keeps its own Caption dropout option.

**Changing settings while a run is paused.** While a run is paused, the settings **Resume** takes from the form are marked **Applies on Resume**: the learning rate (or Adaptive LR's Min and Max), the save schedule, block swap, the picture size and the Samples tab's previews. Everything else trains on as the run began.

**A table view of your captions.** The Captions tab's **View** switch shows a dataset as a table: thumbnail, file name, caption (edit it in place), word count, size and mask, sorted by a click on any column. **Find, Replace & Add** can now add words to the start of every caption (after the trigger word, so it stays first) or the end, with a preview first.

**More in the sidebar.** Under the GPU, VRAM and RAM meters:
- **Disk:** free space on the drive your LoRAs are saved to, red when it's running low.
- **A loss line** on the run card.
- **Background jobs** as meters: captioning, a model download, Image Prep, the Profiler or an extraction.
- **What's loaded on the GPU:** training, Studio's model, the caption model or a helper's previews.

Each one opens its tab on a click.

**Updating:** update Fizgig on your computers, and restart your pods.
