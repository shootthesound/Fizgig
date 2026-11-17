# Fizgig v9.7.4: Gizmo's crop box reworked, AMD fixes, and Automagic v4's real rate

**Gizmo's crop box, reworked.** Choosing an area works the way you'd expect. Drag from any corner in any direction, and the corner you pressed stays put. With a shape locked (1:1, 16:9, 9:16 and the rest), the box grows smoothly within 2% of the shape instead of jumping between a few exact sizes. The box can sit anywhere and reaches every edge, so the bottom of a 1080 frame is no longer out of reach. Drag inside the box to move it, type X / Y / W / H, place it with the arrows or **Largest**, and keep the boxes you use as **Saved boxes**, which follow a video of another size (#350, thanks @CognitiveDiffusion).

**AMD PCs with built-in graphics train on the graphics card.** On a Windows PC with a processor's built-in graphics as well as an AMD card, `Launch Fizgig AMD.bat` never got as far as picking the card, so Fizgig ran on the built-in graphics and stopped at the first step with "invalid kernel file". The launcher now picks the card, and Fizgig picks it itself if nothing else has. Each process also writes one line saying which devices it sees and which it uses, and `Launch Fizgig AMD.bat check` runs a step-by-step diagnostic to paste into an issue (#342, thanks @johnstephendonahue-ux).

**Krea 2 LoKR on 16 GB AMD cards.** A Krea 2 LoKR run with EMA at 0.5 MP could run short at the switch to full size, from memory fragmented by pictures of many shapes. On a Windows card that showed up as steps slowing to a crawl and then a crash. On Linux, AMD now uses expandable memory segments, which fixed the same run in our testing at the same speed. Windows doesn't support those, so on a Windows AMD card a 4-bit plan that only just fits now warns to turn **Weight averaging (EMA)** off, which made that run fit (#341, thanks @scryptio).

**Automagic v4 shows the rate it really uses.** Automagic v4 sets its own learning rate and usually sits at its 2.5e-4 cap, then slows each weight by how sensitive it is, so the flat 2.5e-4 line didn't say what the weights were doing. The run card, the loss chart's rate line and the log now show its working rate: its rate times that slowdown on average. Each epoch line in the log gives both parts. The run card also showed the Learning rate box's value for optimizers that set their own rate; it now shows the live rate (#353, thanks @prookyon).

**Resume Training catches a folder mix-up.** Start now stops with a plain message when the Resume Training box holds the folder that contains the saved states rather than a state itself (it names the newest one to pick), or when the LoRA output folder is a saved state. Before, the run failed with "not a saved training state" (#352, thanks @RandomTurtle).

**Image Prep skips face close-ups not worth making.** A new **Min face width** (400 px in the photo by default) means a photo whose face is smaller than that gets no close-up, as it has too little detail to be worth cropping. That's fewer close-ups to cull by hand. Set it to 0 for the old behaviour.

**Continue says what it does.** Continue on a finished run now says it carries on from the last epoch in the same run, nothing restarts, and the epochs and checkpoints already saved stay as they are.

**To update:** run the updater as usual. On a Windows PC with an AMD card, the fixed launcher comes with the update.
