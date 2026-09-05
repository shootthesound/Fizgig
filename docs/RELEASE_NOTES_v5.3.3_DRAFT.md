<!-- DRAFT — unreleased -->
# Fizgig v5.3.3

A small one for MiniMax H3 training on tight cards.

## The training adapter steps off the card for previews

The training adapter is switched off for every preview render (the preview is the deployment view), but its weights stayed on the card — about 150 MB. On a 24 GB card the int8 plan leaves previews only a few GB, and that 150 MB was enough to tip a clip preview from paging only during the decode into paging on every sampling step (a 4090 report, 5 Sep). The adapter is now parked in system RAM for the render and put back for training; the console says how much came back. Set `FIZGIG_NO_ADAPTER_PARK=1` to keep the old behaviour.
