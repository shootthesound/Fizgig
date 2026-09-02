# Fizgig v5.1.0

A third base precision for MiniMax H3, built by the community, plus the small things
that landed since 5.0.

## MiniMax H3: 4-bit HQQ base precision

Contributed by **[@rintic-13](https://github.com/rintic-13)** (#102), with the streaming
ring reviewed and a bug in it caught by **[@mabseyuk](https://github.com/mabseyuk)**.

Until now an H3 base was either **int8** (the checkpoint's own storage, ~0.17% error,
~21 GB) or **4-bit** NF4 (~9.5% error, ~11 GB). **4-bit HQQ** sits between them: the
same 4-bit codes with a per-group solver that lands at **~6.3% base error — a third closer
to int8** — for **~15 GB** on the pruned checkpoint. Like the other two it streams through
an H2D ring when the card needs block swap, and the ring is verified bit-exact against
classic parking.

It is an **explicit pick** — Auto never chooses it — because it has a real cost: on the
same run it trained at **about half the step speed** of 4-bit NF4 (1.45 vs 2.6 it/s on a
5090), since its de-quantisation runs on the PyTorch path in both forward and backward.
Pick it when the base error matters more than the clock; leave Auto alone otherwise and
**nothing changes for you**.

**To use it:** install the `hqq` package into the venv first, skipping its optional
kernel build —

```
Windows:  set DISABLE_CUDA=1 && venv\Scripts\pip install hqq==0.2.8.post1
Linux:    DISABLE_CUDA=1 venv/bin/pip install hqq==0.2.8.post1
```

— then choose **4-bit HQQ** under Base precision on the Training tab (or
`--base_quant hqq` on the command line). Picking it without the package installed stops at
load with the install line, not mid-run.

## Checkpoint to LoRA, from the fine-tune cards and on Linux

The **Checkpoint to LoRA** utility that turns a fine-tuned checkpoint into an ordinary
shareable LoRA now has a button on both fine-tune cards, and a Linux/pod launcher
(`run_diff_to_lora.sh`, twin of the `.bat`) so it no longer needs invoking by hand on a
pod.

Easy to miss, now said everywhere it matters: a **fine-tuned checkpoint can be set as the
base model in Preferences**, and LoRAs trained on top of it.

## Notes from the field

- **12 GB cards, Windows:** previews park the model to system RAM, which spikes system
  commit — a fixed 4 GB paging file dies with error 1455 and a message that names
  nothing. Set the paging file to **system-managed**. Reported and confirmed on a 12 GB
  RTX 5070 by **David Maybank**.
- **Krea 2 LoKR previews need more than 16 GB** — the render runs on the resident training
  model with the LoKR net live. The trainer now says so up front, directly above where a
  16 GB card would otherwise stall.
