# Fizgig v5.3.1

A fix for Repair Studio on MiniMax H3: library peeks and history views work again.

## Library peeks and history views

In 5.3.0, clicking a block's ● to see the clip with that block off, clicking a history thumbnail, or choosing View from its right-click menu did nothing: the clip failed to load behind the scenes and the status line went back to normal a second later. The pinned baseline was affected the same way. Fixed — every one of them shows its clip in the tweaked pane and the player as intended. Renders, the library build itself and everything else were unaffected.
