# Fizgig v5.4.0

MiniMax H3 training gets faster three ways, and every clip in your dataset now teaches a sharp still as well as its motion.

## The int8 base is about 12% faster per step 

The fused W8A16 Triton kernel by **[@rintic-13](https://github.com/rintic-13)** shipped in 4.3.2 and did not always run when it should. On a 5090 at 0.25 MP that is 1.06 s a step down to 0.94 on the same run. There is nothing to switch on.

## Dave Maybank's backward kernel, on by default

**[@mabseyuk](https://github.com/mabseyuk)** wrote the companion: a fused backward for the int8 base that computes the input gradient without materialising a bf16 copy of each weight. Measured on the real ConvRot shapes, every element lands within one bf16 ulp of the eager backward, and against an fp32 reference it is marginally closer. On top of the forward kernel it is worth a few percent per step: 2–3% on a 5090 at 0.25 MP, 2% at 0.5 MP, 8% on a simulated 24 GB card streaming 34 blocks. It runs alongside the forward kernel on every int8 plan; `FIZGIG_NO_TRITON_W8A16_BACKWARD=1` turns it off, `FIZGIG_NO_TRITON_W8A16=1` turns both off.

Both kernels now tune once per token-count bucket instead of once per exact token count. Keyed on the exact count, a first epoch spent minutes re-tuning for every caption length and kept re-tuning as caption dropout shifted lengths; now the tuning is a couple of dozen events in the first steps.

## TREAD token routing — on by default, one tick to turn off

TREAD is from [Krause, Phan, Hu and Ommer, *TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training* (arXiv 2501.04765)](https://arxiv.org/abs/2501.04765). On every clip step a random half of the video tokens leaves the sequence at block 2 and rejoins at block 47 in exactly the state it left in, so 45 of the 50 blocks process half the tokens. Clip steps get markedly faster, and that is the smaller half of the point.

The paper's finding is that routing makes a diffusion model *learn better*, not just cheaper. The late blocks receive a sequence in which half the tokens carry only their early-block state, so they cannot lean on a fully processed neighbour for every position and have to build the prediction from less. That works like a regulariser on the representation: in the paper's experiments the routed model converges in a fraction of the steps of the same model trained plainly and ends at better quality, with nothing added to the architecture and nothing changed at inference. That last part matters here: the trained LoRA is an ordinary LoRA, previews never route, and ComfyUI never knows it happened.

Photos — and the clip stills below — always run in full: a still has no neighbouring frames to lean on, and it is where the sharp identity signal lives. In our A/B the split version — routed clips, untouched stills — was the one that held up, so it is the only version. It is a tickbox on the Training tab, on in every preset for LoRA runs: untick **TREAD token routing** for a plain run, or to compare.

## Every clip also trains its sharpest face frame as a photo — optional, on by default

When clips are cached, every frame is scored for focus and the sharpest one that shows a face is picked — the score is taken on the face itself, so subject motion blur decides, not background texture — and encoded as a still. It then trains on a step of its own with the clip's caption: a sharp second look at every subject, at no cost to the clip step. Frame-filling close-ups are handled. It is the tickbox **Also train each clip's sharpest face frame as a photo** on the Training tab, on in every preset except Style; untick it and clips train as clips only. Clips cached before this was on use frame 0 until they are re-cached; the cache step at your next launch adds the picks to just those clips and leaves everything else alone.

## Optimised Likeness Learning always confines clips

The "Restrict video to likeness blocks" sub-tick is gone: whenever Optimised Likeness Learning is on, clips train the identity blocks (20–49) alongside photos, in LoRA and fine-tune runs alike. In our tests that trains video just as well and makes clip steps far lighter on VRAM. Untick likeness mode to train video on the whole model, which is what style and scene training already do.

## Smaller things

- **Learn identity from my dataset** (reference distillation) has moved into Other Options.
- Multi Concept's hint and README bullet no longer claim it changes caption dropout; it stopped doing that after the August A/B where one folder with dropout beat two without.
- The TREAD and clip-still hints are one line each; the detail lives in the README's MiniMax section, under "Training-tab controls worth knowing".

## Thanks

To **[@rintic-13](https://github.com/rintic-13)** for the forward kernel, and to **[@mabseyuk](https://github.com/mabseyuk)** for the backward kernel, the parity harness that came with it, and the patience while it waited its turn.
