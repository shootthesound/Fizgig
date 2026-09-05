# Fizgig v5.3.1

A fix for Repair Studio on MiniMax H3: library peeks and history views work again.

## Library peeks and history views

In 5.3.0, clicking a block's ● to see the clip with that block off, clicking a history thumbnail, or choosing View from its right-click menu did nothing: the clip failed to load behind the scenes and the status line went back to normal a second later. The pinned baseline was affected the same way. Fixed — and they now do more than show the clip: clicking a library chip or a history entry sets the sliders and ticks to the state that made that clip, so what you see is what Save writes. The library serves it instantly as before. Renders and everything else were unaffected.

## The library builds banks of five

The library used to render every block switched off on its own — 52 entries, three and a half minutes at ⅔ size, and on MiniMax a single block rarely shows. It now renders **banks of five blocks** switched off: 0–4, 4–8, 8–12 and so on to 44–49, each bank sharing one block with the next, so a feature that two neighbouring banks both lose sits in the block they share. Twelve entries, under a minute at ⅔ size, at whatever size and steps the Clip row says. Refiners are never in a bank and stay on unless you turn one off. The chips above the sliders replace the dots beside them: hover for a thumbnail, click and the sliders move to that state and the clip appears.
