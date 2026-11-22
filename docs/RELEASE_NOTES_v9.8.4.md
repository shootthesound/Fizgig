# Fizgig v9.8.4: Studio switches LoRAs in a second, and AMD finds its kernel database

**Switching the primary LoRA in Studio no longer reloads the model.** Choose another primary and press Render: the old LoRA comes off and the new one goes on the model already loaded, in a second or so, where it used to reload the whole model (minutes on a large one such as MOSS-Image). The render is the same as a fresh load of that LoRA. A LoRA that can't load (one for another model, say) leaves the old one on and says why. MiniMax H3 video and Frankenstein mode still load again. ([#394](https://github.com/shootthesound/Fizgig/issues/394), thanks @tsk5cb-cmyk)

**AMD: MIOpen finds its packaged kernel database.** Both AMD launchers pointed MIOpen at a folder with no database in it, so it couldn't use its packaged kernel tuning. The Linux launcher now points at the database ([#384](https://github.com/shootthesound/Fizgig/pull/384), thanks @xGnome), and so does Launch Fizgig AMD.bat on Windows ([#389](https://github.com/shootthesound/Fizgig/issues/389), thanks @dkspwndj for the report).

**A GGUF file no longer slows Fizgig down.** A GGUF quant in the secondary captioning model slot was read in full at every page load, about 15 seconds for a 20 GB file; the captioner now takes only `.safetensors` files ([#388](https://github.com/shootthesound/Fizgig/pull/388), thanks @spaghettified26). Every other place that reads a model file's header refuses a file like that from its first eight bytes too, where it used to read gigabytes before failing.

**To update:** run the updater as usual.
