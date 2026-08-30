# MiniMax H3 — how much of training happens at LOW NOISE ("Low-noise training" on the tab).
#
# The dial is the SHARE, not a shift number. Same underlying knob, stated as the thing that
# actually matters. Training draws u uniform and maps it through
#
#     sigma = shift * u / (1 + (shift - 1) * u)
#
# so the fraction of steps landing below a threshold T is P = T / (shift*(1 - T) + T). At
# T = 0.5 that inverts to exactly
#
#     shift = (1 - P) / P
#
# — no solver, no table, no resolution term. Reference points: 50% -> shift 1 (unshifted
# uniform), 22% -> 3.5, 7.7% -> 12 (H3's own video default).
#
# DELIBERATELY UNCAPPED and deliberately independent of megapixels. The previous MP-keyed
# version derived its numbers from a sqrt-token transfer rule that has never been validated on
# H3, and capping the list to that rule's answers ruled out settings worth trying.
MINIMAX_LOWNOISE_SIGMA = 0.5     # "low noise" = the cleaner half of the sigma range


def minimax_lownoise_to_shift(pct):
    """Share of steps below sigma 0.5 (percent) -> the shift that produces it. None if unusable.

    0 and 100 are the asymptotes, not values: 0% needs an infinite shift and 100% a shift of 0,
    and neither is a schedule. Everything strictly between them is allowed."""
    try:
        p = float(str(pct).strip().rstrip("%")) / 100.0
    except (TypeError, ValueError):
        return None
    if not (0.0 < p < 1.0):
        return None
    return (1.0 - p) / p


def minimax_highnoise_lr(pct):
    """'Medium to High LR adjustment' (percent) -> a plain multiplier. None if unusable.

    Applies to steps drawn ABOVE sigma 0.5 — the same threshold the low-noise box is defined
    against, so the two controls always agree about where the boundary is. 100 means unchanged.
    0 is allowed and means those steps train nothing; the ceiling is 100 because raising the LR
    for the noisy end is a different experiment from damping it, and one dial should do one thing.
    """
    try:
        p = float(str(pct).strip().rstrip("%")) / 100.0
    except (TypeError, ValueError):
        return None
    return None if not (0.0 <= p <= 1.0) else p


# The logit-normal ("mid-concentrated") variant of the above is GONE, deliberately.
#
# It bunched training around the middle and thinned BOTH tails, and the tail it was quietly
# deleting is the high-noise end where pose and composition are decided — at 60% low noise it left
# 0.7% of a run above sigma 0.9. A 20-step render spends most of its steps there, so the LoRA was
# being asked to hold structure it had barely trained on: fine under a 4-step Turbo workflow,
# soft or distorted without one.
#
# Tested directly (Peter, 14 Aug): the Likeness preset with mid-concentrated OFF works at strength
# 1.0 without Turbo, and likeness did not suffer — so it was not buying what it was supposed to buy
# either. The 60% share was never the problem.


def minimax_shift_to_lownoise(shift):
    """The inverse — shows an existing shift (e.g. from an older preset) as a percentage."""
    try:
        s = float(shift)
    except (TypeError, ValueError):
        return None
    return None if s <= 0 else 100.0 / (1.0 + s)


# MiniMax H3 — which of the 50 DiT blocks to train ("Blocks to Train" on the Training tab).
#
# THIS IS A SEARCH, not a recommendation, and the labels say so. H3 is 50 identical blocks with
# no published map. The ranges come from the proportions of Klein's empirically-built map
# (composition in the first ~30% of depth, identity ~30-75%, fine detail ~60-100%) scaled to 50
# blocks — an analogy between two architectures, not a measurement of this one.
#
# READ EVERY RESULT AS: "was this subset sufficient?" A range that trains a good likeness means
# its COMPLEMENT was not needed — whichever end that turns out to be. There is no option here
# that disproves the idea; a surprising winner just relocates the answer. (An earlier revision
# labelled the front half a "control, expected worse". That was wrong: a good front-half result
# would mean the back half is droppable, which is the outcome we are hunting, not a refutation.)
#
# Why dropping blocks may HELP rather than merely cost less: a block that carries no identity
# still receives gradient, and what is left for it to learn is the dataset's background, framing
# and lighting. Excluding it removes capacity that would otherwise go into memorising the set.
#
# What is measured: per-block ||dW|| on two finished H3 LoRAs is nearly FLAT (3x quietest to
# loudest against a 2% flat expectation, thirds within 28/34/38), and the quietest blocks do not
# agree between runs. Weight movement offers no map either way — which is why the only way to
# learn anything here is to train a range and compare.
# The box is EDITABLE: these are jumping-off points, and anything the trainer's parser accepts
# can be typed instead — "3-12, 14-15, 22, 31-33". The label separator is "·" and not "-" or a
# plain space, because both of those appear inside a block spec; splitting on them would turn
# "3-12, 14" into "3-12," and train the wrong set without ever complaining.
# MiniMax H3 — "Training Structure" on the Training tab. One decision, two numbers behind it.
#
#   pct  = share of steps trained below sigma 0.5 (the clean end, where detail and identity live).
#          Converted to the trainer's --shift by minimax_lownoise_to_shift.
#   lr   = what the steps ABOVE that threshold do to the learning rate, as a percentage.
#
# Both presets recommend 100 — the noisy steps train at full rate — because that is the only
# setting anything has been measured at. The dial exists because dropping the clean-end share
# makes high-noise steps the MAJORITY (92% at the model's own schedule against 40% here), and the
# worry was that they would swamp the few clean-end steps carrying identity.
#
# Measured across FIVE datasets (Peter, 14 Aug), at BOTH densities and at 0% and 100%: nothing
# corrupts in any of the four combinations, and 100% has visibly better face SHAPE every time.
# That fits — shape is settled early, at high noise, while skin and texture come late — so damping
# costs geometry rather than buying stability, including at 8% clean-end where the noisy end is
# 92% of the run and the swamping worry should have been strongest. It was not. Nothing is damped
# by default.
#
# The box stays because it is the SAFE way to bias a run toward surface detail. Mid-concentrated
# tried to do that by changing which noise levels were SAMPLED, which took the adapter
# off-distribution and distorted at 20 steps. This changes only how much is learned from them —
# the schedule the model sees is untouched, which is why 0% renders cleanly at either density. A
# skin-texture LoRA is the obvious use.
#
# The 8% is not a guess. ai-toolkit's H3 entry overrides its global 'sigmoid' timestep type with
# 'shift' (ui/src/app/jobs/new/options.tsx) against a scheduler at the model's RELEASED video flow
# shift of 12 (minimax_h3.py + packing.py). A shifted-uniform draw at shift 12 puts 1/13 = 7.7% of
# steps below sigma 0.5. It is the schedule the model's own flow shift implies — not a published
# statement of what MiniMax ran in pre-training, which is why the label says the former.
# There is deliberately no "style" setting between these two. Style lives at the CLEAN end, not
# the noisy one — Fizgig's own Klein work established that, extracting at three timestep ranges
# and finding style concentrated in the late/clean band. Brushwork, palette and grain are surface
# properties, so a style LoRA wants the same density a likeness one does; what distinguishes it is
# often rank and LR, not the noise schedule.
# An earlier revision of this shipped a "Balanced / style — 25%" option built by treating style as
# composition and pushing AWAY from the clean end. That was backwards, and a mislabelled setting is
# worse than a missing one.
MINIMAX_STRUCTURE_OPTIONS = {
    "Likeness and Style — 60% clean-end": (60, 100),
    "Model default, movement — 8% clean-end": (8, 100),
    "Custom": None,
}
MINIMAX_STRUCTURE_DESC = {
    "Likeness and Style — 60% clean-end":
        "Most of the run on nearly-clean images. Skin, hair and identity are learned there — and "
        "so is style, which is a surface property rather than a compositional one. The tuned "
        "default for stills.",
    "Model default, movement — 8% clean-end":
        "The schedule H3's own flow shift implies, and what the reference trainer uses. Weighted "
        "to movement and composition rather than fine detail.",
    "Custom":
        "Type your own share. Below ~50% the high-noise steps become the majority, which is what "
        "the LR adjustment beside this is for.",
}
MINIMAX_STRUCTURE_DEFAULT = "Likeness and Style — 60% clean-end"

MINIMAX_BLOCK_OPTIONS = [
    "all · every block (50 of 50)",
    "10-49 · skip the first 10",
    "14-37 · middle band",
    "25-49 · back half",
    "0-24 · front half",
]


MINIMAX_NUM_BLOCKS = 50          # H3's DiT block count (MiniMaxH3Config.num_layers)

# Optimised Likeness Learning — the block set photo steps train when the checkbox is on (clips
# always train the full model). 20-49 is the measured recipe (19 Aug): the visual-exclusive band
# 20-26 adds pose/likeness stability, identity lives 27-49, and the front trunk 0-19 is where
# photo gradients deform anatomy. One place to tweak as the add-back ladder refines the figures.
MINIMAX_LIKENESS_BLOCKS = "20-49"

# Voice routing — the block set audio-only steps train. 34-49 per the block map (audio core
# 38-48 peak 41-42, shoulder 34-37) and Peter's A/B (24 Aug): audio-only trained at 34-49 is
# clean; at 20-49 the audio training corrupted the visual blocks. Clips still train the full
# model (pending the same test for video).
MINIMAX_AUDIO_BLOCKS = "34-49"

# Base Precision — the label the user sees, and the --base_quant value it sends. Auto plans the
# quantisation and the block-swap count together (see plan_base_quant in minimax/trainer.py);
# an explicit pick is never overridden, the swap plan is built around it instead.
MINIMAX_BASE_QUANT_OPTIONS = [
    "Auto (recommended)",
    "int8 · most accurate, needs ~30 GB free",
    "4-bit · fits smaller cards",
]


def minimax_base_quant(raw):
    """Dropdown label -> the --base_quant value. Anything unrecognised falls back to auto."""
    s = str(raw or "").split("·")[0].strip().lower()
    if s.startswith("int8"):
        return "int8"
    if s.startswith("4-bit") or s.startswith("nf4"):
        return "nf4"
    return "auto"


def minimax_block_spec(raw):
    """The block selection out of a dropdown label OR a hand-typed spec. "" -> "all"."""
    return str(raw or "").split("·")[0].strip() or "all"


# Training Base — which H3 fine-tune the LoRA trains against. fl2va (first/last-frame) is the
# ordinary model; ref2va is the Reference-to-Video fine-tune ComfyUI's r2v workflow runs. Same
# architecture and shapes, so the trainer takes either — a LoRA is most faithful on the base it
# trained against, which is the point of the choice. Deliberately NOT preset-affected: the var
# lives outside self.entries and is never collected, so presets/last-train can't flip it.
MINIMAX_TRAIN_BASE_OPTIONS = [
    "First/last frame (fl2va) — standard",
    "Reference (ref2va)",
]


def minimax_train_base(raw):
    """Dropdown label -> canonical base key. Anything unrecognised is the fl2va default."""
    return "ref2va" if "ref2va" in str(raw or "").lower() else "fl2va"
