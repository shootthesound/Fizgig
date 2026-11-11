# Fizgig v9.6.4: Likeness optimization for MiniMax H3 Image

**MiniMax H3 Image has a new Likeness optimization**, and in our tests it improves likeness dramatically. It's on in the Standard preset: load **✨ MiniMax H3 Image Standard** to pick it up. It replaces H3 Image's Clean-end share box.

It puts three quarters of training on the last of the model's 6 steps, the nearly finished picture where skin, eyes and fine detail are made, for the first 8 epochs. After that it alternates between half and three quarters, so the earlier steps that set pose and layout keep up. Untick it on the Training tab (Training Parameters) for H3 Image's own even share. Style, Edit and Slider leave it off.

It will be updated over the coming days with more ways to customise it.

**Also in this release:**
- **The loss graph's learning-rate line no longer disappears** on long runs (past about 4,000 steps).
- **MOSS-Image on an older PyTorch** keeps its earlier compile memory use. An install updated with `git pull` but still on PyTorch 2.10 could spill into system RAM and train many times slower (#324). Running the updater for PyTorch 2.15 is still the faster option.
