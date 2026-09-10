# Fizgig v5.6.0

MiniMax H3 training takes a big step: a large quality improvement in both the visual and the audio results, a much smoother climb through the epochs, and training steps about 30% faster. Nothing to set. It is the new default on every H3 LoRA run.

## MiniMax H3: better results, smoother convergence, faster steps

Two small changes to what a LoRA trains, both on by default under Optimised Likeness Learning: the LoRA no longer touches the model's text token refiner, the part that decides how a prompt is read, and the backward pass now stops at the likeness window instead of running the whole model. What that gives you:

- **Sharper, cleaner output, faces and voices alike.** On a same-seed comparison the likeness climbed steadily and held where it used to wobble, and a voice was recognisable by epoch 4 where it used to take until the low teens.
- **Previews that evolve smoothly from epoch to epoch.** The same scene refining rather than re-laying itself each time, so any epoch is a fair checkpoint and the Training Run Visualiser reads as a true scrub of the run.
- **Training steps about 30% faster.** On int8 the same run went from 2.6 to 3.4 steps a second, 23% less time per step; 4-bit gains a little more at 27%. Previews are unchanged, so a run with a big clip preview every epoch sees less of it; a run with previews off, or stills, sees all of it.

Trigger words work exactly as before. If you want the refiner trained for a comparison of your own, there is a tick in Other Options: off in every preset, recommended off. On the command line, `--train_token_refiner` and `--likeness_full_backward` restore the previous behaviour.

## Krea 2 previews on 16 GB cards

On 16 GB cards running the 4-bit base, the in-training preview could push past VRAM at the decode and leave the rest of the run slow. On cards under 20 GB Fizgig now parks the training model for the decode and caps the preview canvas at 768 px, with `[preview-vram]` lines in the console showing the headroom at each stage. Bigger cards are unchanged. Reported by **[@Linkram](https://github.com/Linkram)** (#123), thank you.

## Training tab: shorter hints

The explanations under the MiniMax training controls have been rewritten to be read at a glance.
