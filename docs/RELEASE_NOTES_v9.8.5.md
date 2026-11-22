# Fizgig v9.8.5: Studio swaps LoRAs for the queue and other tabs too, and the Windows AMD launcher's MIOpen path is back

**Studio's queue swaps LoRAs instead of reloading.** A queued job whose only difference is the primary LoRA now swaps it on the model already loaded, as Render does since 9.8.4, and shows Loading LoRA meanwhile. Afterwards your next Render swaps your own LoRA back on in a second or so instead of loading the model again. A job with a LoRA that can't load fails with the reason. A different model, donor or settings still loads again, and so do MiniMax H3 video and Frankenstein mode.

**Studio in two tabs or windows.** Studio holds one model at a time, so a second tab that loads its own setup replaces the first tab's. The first tab now says another tab has loaded over it, and its Render loads its own setup again, where before it gave no sign and could render on the other tab's LoRA. Switching or unloading the primary LoRA in one tab counts as a new load in the others too. ([#395](https://github.com/shootthesound/Fizgig/issues/395), thanks @tsk5cb-cmyk)

**Frankenstein mode needs 32 GB of system RAM**, not the 48 to 64 GB the 9.8.3 notes said. The higher figures counted model-file pages Windows can release.

**Windows AMD: MIOpen's path is back where the Windows wheels keep it.** 9.8.4 moved Launch Fizgig AMD.bat's MIOpen database path to the Linux folder layout, which doesn't exist on Windows, so the launcher set no path at all. It points at `_rocm_sdk_devel\bin` again, as before 9.8.4. ([#389](https://github.com/shootthesound/Fizgig/issues/389), thanks @scryptio)

**AMD: `FIZGIG_MIOPEN=0` turns MIOpen off for convolutions**, so they use PyTorch's own kernels, as ComfyUI does by default on RDNA3 and newer cards. It's off unless you set it, and each stage's log says when it's on. It's here to test a freeze during a latent recache on an RX 7900 XTX ([#389](https://github.com/shootthesound/Fizgig/issues/389), thanks @dkspwndj for the detailed report).

**To update:** run the updater as usual.
