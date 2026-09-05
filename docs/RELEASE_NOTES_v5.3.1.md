# Fizgig v5.3.1

A fix for Repair Studio on MiniMax H3: library peeks and history views work again.

## Library peeks and history views

In 5.3.0, clicking a block's ● to see the clip with that block off, clicking a history thumbnail, or choosing View from its right-click menu did nothing: the clip failed to load behind the scenes and the status line went back to normal a second later. The pinned baseline was affected the same way. Fixed — and they now do more than show the clip: clicking a dot or a history entry sets the sliders and ticks to the state that made that clip (every other block at its default, that one unticked), so what you see is what Save writes. The library serves it instantly as before. Renders, the library build itself and everything else were unaffected.
