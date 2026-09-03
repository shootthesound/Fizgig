# Fizgig v5.3.0 — DRAFT (unreleased)

## MiniMax H3: 4-bit HQQ moves to group 8 — a third less base error at the same size

HQQ 4-bit now quantises in groups of 8 instead of 16, taking the frozen base from ~6.3% to ~4.8% error against the int8 reference. Plain group 8 would have doubled the per-group overhead to int8's own footprint, so the per-group scales and zeros are themselves stored as 8-bit codes with one affine per output row — measured at 4.83% versus plain group 8's 4.80%, at exactly the group-16 footprint (0.75 bytes per weight, ~15 GB resident on the pruned checkpoint). Group size and the 6% cost on a 16 GB card are **[@rintic-13](https://github.com/rintic-13)**'s numbers (#102).

Also corrected in the docs: HQQ's speed cost is a big-card statement. On a plan that streams blocks — the 12–24 GB tiers it exists for — the dequant hides behind the PCIe transfers and HQQ runs at NF4's speed; only on a large card with nothing streamed does it show as roughly half NF4's step speed.

## Multi Concept no longer switches identity-learn on

Ticking Multi Concept now sets caption dropout to 0.10 (strong) and nothing else. It used to switch reference distillation on as well, with its references and identity-first phase; in our runs that wasn't what held two subjects apart — the trigger words do that work — and a data-layout tick quietly enabling a 21 GB-model experiment was the wrong shape. Identity-learn stays its own deliberate tick.
