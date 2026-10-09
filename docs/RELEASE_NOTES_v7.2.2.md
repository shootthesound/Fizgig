# Fizgig v7.2.2: maintenance

The Samples tab keeps each model's settings through everything else that touches it, and training previews no longer pause for minutes after a start or resume when Compile Blocks is on.

## Fixes

- **Each model's Samples settings stay its own.** A few things outside the Samples tab could still save one model's CFG, Steps or preview size under another:
  - **The training queue.** Loading a queued run of another model saved that run's values under the model that was on screen.
  - **A run's settled preview size.** When previews stepped down to a size that fits, the size was saved to whichever model was on screen at the time.
  - **Old presets.** A preset saved by an older Fizgig could still change the Samples values.
  - **MiniMax H3.** A typed Steps value of 6 went back to 20.
- **Turbo strength.** Typing a new Turbo strength and pressing Start straight away now switches Steps and CFG first.
- **Compile Blocks: no long pause at the first preview.** With Compile Blocks on, the first preview after starting or resuming a run spent several minutes compiling (the console sat on a `__triton_launcher` line). Previews now run uncompiled and take about the same time every epoch; training keeps its compiled speed.

To update, run `update_fizgig.bat` (`update_fizgig_rocm.bat` on AMD). Pods pick up the update on their next start.
