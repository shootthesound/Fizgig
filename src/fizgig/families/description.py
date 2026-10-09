"""FamilyDescription: everything Fizgig needs to know about one model family, in one object.

Adding a family used to mean threading it through the GUI by hand (predicates, if-chains, Preferences
blocks, preset dicts, command builders). A description holds the family's facts once; the GUI's generic
paths read it - its Preferences section, Base Model rows, Multi Concept, Samples wording and queue summary
included, so a new family plugs in without per-family GUI code.

Every value that came from outside Fizgig carries its source (`source=` fields), so a later reader can
re-check it when the upstream model or a speed LoRA moves on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# the Samples tab's long-standing negative prompt, the default for the families that shipped with it
GENERAL_NEGATIVE = ("blurry, low detail, noisy, washed out, oversaturated, distorted anatomy, extra limbs, duplicate "
                    "objects, text, watermark, logo, frame, cropped subject, flat lighting, muddy colors")


@dataclass(frozen=True)
class ModelFile:
    """One file the user points Fizgig at in Preferences."""
    pref_key: str                     # prefs.json key, e.g. "qwen21_dit"
    label: str                        # Preferences row label
    required: bool = True             # training cannot start without it
    repo: str = ""                    # Hugging Face repo for the Download link
    path: str = ""                    # file path inside the repo
    size_gb: float = 0.0
    note: str = ""                    # one plain line shown under the row
    local_name: str = ""              # name in models/ when the repo's own is generic (diffusion_pytorch_model...)
    role: str = ""                    # "dit" | "vae" | "text_encoder" | "training_adapter" | "speed_lora" |
    #                                   "preview_dit" (a separate model the workbench previews on, e.g. a distilled
    #                                   checkpoint; see preview_checkpoint_sampling) | ""
    # Preferences row extras: the longer text under the row (else `note`), the Download link's label and the line
    # beside it (else "~N GB - repo -> file"), and a second link to another build of the same file
    hint: str = ""
    download_label: str = ""
    download_note: str = ""
    alt_repo: str = ""
    alt_path: str = ""
    alt_label: str = ""
    alt_note: str = ""
    gated: bool = False               # the repo needs an accepted licence + a Hugging Face token to download
    # the download button fetches it only with the section's optional tick (description.fetch_optional_label);
    # None = not required -> optional
    fetch_optional: Optional[bool] = None
    # a file added after most users set the family up: its line in the one-time "new model files" popup (shown when
    # the family's DiT is set but this file is not; description.announce_intro opens it). "" = never announced
    announce: str = ""
    # a part that ships inside another model file (a single-file checkpoint's VAE / text encoders): the pref key of
    # that file, used when this row is left empty - the row stays as an optional override (e.g. a fixed VAE)
    inside: str = ""

    @property
    def fetch_is_optional(self) -> bool:
        return (not self.required) if self.fetch_optional is None else bool(self.fetch_optional)


@dataclass(frozen=True)
class SamplingSettings:
    """A complete, citable way to sample the family: steps, CFG, sampler, schedule."""
    name: str
    steps: int
    cfg: float
    sampler: str = "euler"
    scheduler: str = "simple"
    sigmas: Optional[tuple] = None    # explicit schedule when the model needs one
    options: tuple = ()               # driver-specific sampler options as (name, value) pairs
    negative_prompt: bool = False     # whether a negative prompt does anything at this CFG
    note: str = ""
    source: str = ""


@dataclass(frozen=True)
class SpeedLoRA:
    """A Turbo / Lightning / distill LoRA for the family, with the settings it actually wants."""
    name: str
    repo: str
    file: str
    pairs_with: str                   # which base it was trained against
    strength: float
    settings: SamplingSettings
    load_unmerged: bool = False       # merging into the weights loses part of it
    pref_key: str = ""                # the model file (Preferences row) holding it
    community_settings: tuple = ()    # (description, source) pairs: how people actually use it
    caveats: tuple = ()
    source: str = ""


@dataclass(frozen=True)
class LoRAFormat:
    """How this family's LoRA keys are written so ComfyUI (and diffusers) load every module."""
    key_template: str                 # e.g. "transformer.transformer_blocks.{block}.{module}.{ab}.weight"
    down: str                         # "lora_A" / "lora_down"
    up: str                           # "lora_B" / "lora_up"
    block_modules: tuple              # per-block Linears the LoRA targets
    alpha_key: str = "{prefix}.alpha"
    kohya: bool = False               # True = the lora_unet_ convention used by Klein/Krea 2/H3
    file_prefix: str = ""             # what precedes a module path in every key, e.g. "transformer."
    # a kohya family's LoKR keys: LyCORIS's 'diffusion_model.<module path>' by default (ComfyUI's own names on Klein,
    # Krea 2, H3, Anima); True writes them on the LoRA's lora_unet_ stems instead, for a model loaded under other
    # names than ComfyUI's (SDXL: diffusers in Fizgig, LDM in ComfyUI)
    lokr_kohya_stems: bool = False
    note: str = ""
    source: str = ""

    def module_of(self, key: str) -> Optional[str]:
        """Dotted module path of a down-weight key ('transformer.modulation.1.lora_A.weight' -> 'modulation.1'),
        None for any other key."""
        tail = f".{self.down}.weight"
        if not (key.startswith(self.file_prefix) and key.endswith(tail)):
            return None
        return key[len(self.file_prefix):-len(tail)]

    def key(self, block: int, module: str, which: str) -> str:
        """which: "down" or "up"."""
        ab = self.down if which == "down" else self.up
        return self.key_template.format(block=block, module=module, ab=ab)


@dataclass(frozen=True)
class ClipSpec:
    """What a video family's clips must be (descriptions with "clip" in media). The dataset decodes and checks every
    clip against it, and the launch refuses an off-spec clip with the reason before anything caches
    (families/clips.py) - Fizgig never converts one."""
    fps: int = 24
    frame_step: int = 17              # frame counts frame_offset + n * frame_step (H3: 17n + 5), up to max_frames
    frame_offset: int = 5
    max_frames: int = 124
    edge_multiple: int = 32           # width and height
    audio_rate: int = 32000
    audio_channels: int = 2
    mute_suffix: str = "_mute"        # a clip named ..._mute trains its pictures only
    note: str = ""


@dataclass(frozen=True)
class FamilyOption:
    """A family's own control on the Training (or Samples) tab: a dropdown, an entry or a tick whose value becomes
    launch tokens. A token is "name=value" (the driver's --family_option), "--flag=value" or "--flag" (a trainer
    flag), or "aux:name=value" (the cache stages' --aux). "{}" in a token is the entry's text; "pref:KEY" in a value
    is that Preferences file's path. A family with clips, sound or several bases (H3) declares what has no generic
    home this way, and a future family declares its own the same way."""
    key: str                          # where the GUI keeps the value (and the old setting it maps, via `setting`)
    label: str
    kind: str = "choice"              # "choice" | "entry" | "check" | "fixed" (always sent, no control)
    choices: tuple = ()               # choice: (label, tokens) pairs, the first the default
    tokens: str = ""                  # entry: tokens with "{}" for the text (sent only when the text is set);
    #                                   check: tokens sent when ticked
    default: str = ""                 # entry text / "1" for a ticked check; a choice defaults to its first label
    hint: str = ""                    # the grey line under the control
    choice_hints: tuple = ()          # (label, amber line) shown while that choice is picked
    choice_notes: tuple = ()          # (label, grey line) shown in place of the hint while that choice is picked
    choice_values: tuple = ()         # choice: the old setting's value each choice stands for, in choice order
    #                                   ("60", "8", ""); "" marks the custom choice any other value lands on
    tab: str = "training"             # "training" | "samples" | "model" (the Training Base row under the Base
    #                                   Model picker: never in a preset, carried by Last Train and the queue)
    # training tab: "" = Training Parameters (under Network Type), "after" = Training Parameters below the "Read at
    # launch" line, "other" = Other Options; a fine-tune option (mode "finetune") sits in the fine-tune card
    section: str = ""
    inline: bool = False              # on the previous option's row, after its control (the label as plain text)
    compact: bool = False             # label and control side by side across the row, not in the label column
    width: int = 0                    # the control's width in characters (0 = sized to its choices)
    suffix: str = ""                  # a grey note beside the control ("% of steps")
    hint_indent: int = 0              # extra left indent of the hint, in pixels
    pady: tuple = (8, 0)              # the control row's vertical padding
    hint_pady: tuple = (0, 4)         # the hint's vertical padding
    always_shown: bool = False        # `requires` decides what is SENT; the row shows regardless
    # while `requires` is not met the control greys out and shows (required choice label, hint, side note)
    unmet_notes: tuple = ()
    counts_blocks: int = 0            # a block-spec entry: the side note counts the blocks it means (of this many)
    setting: str = ""                 # the old settings key this option's value is read from on first use
    show_if_media: str = ""           # shown only when the dataset holds this media kind ("clip", "voice")
    mixed_only: bool = False          # shown only when the dataset mixes voice with photos or clips
    requires: str = ""                # shown (and sent) only while this check is ticked ("KEY") or this choice is
    #                                   picked ("KEY=LABEL", matched as pick() matches)
    suggestions: tuple = ()           # entry: ready-made values offered in an editable dropdown ("6-49 · why")
    mode: str = ""                    # "finetune": shown and sent for a fine-tune only; "lora": for a LoRA only
    # a queued run's one-line summary shows this while the option is away from its default ("{}" = its value;
    # a check shows it while ticked); "" = never shown
    summary: str = ""

    def choice_labels(self):
        return [c[0] for c in self.choices]

    def pick(self, value):
        """The choice label for a stored value: itself, else the label that contains it (an old setting's value -
        "ref2va" -> "Reference (ref2va)") or that it starts with, else the first (the default)."""
        v = str(value or "").strip()
        labels = self.choice_labels()
        if v in labels:
            return v
        if self.choice_values and v:
            for lab, cv in zip(labels, self.choice_values):
                if cv and v == cv:
                    return lab
            for lab, cv in zip(labels, self.choice_values):
                if cv == "":
                    return lab           # an old value no choice names: the custom choice
        low = v.lower()
        for lab in labels:
            if low and (low in lab.lower() or lab.lower().startswith(low.split(" ")[0])):
                return lab
        return labels[0] if labels else ""

    def resolve(self, value):
        """The tokens this option sends for `value` (a choice label, entry text or "1"/"" for a check)."""
        if self.kind == "choice":
            return dict(self.choices).get(self.pick(value), "").split()
        if self.kind == "fixed":
            return self.tokens.split()
        if self.kind == "check":
            return self.tokens.split() if str(value) in ("1", "True", "true") else []
        import re
        text = str(value or "").split("·")[0].strip()          # a suggestion's label: its value before the "·"
        text = re.sub(r"\s*\([^()]*\)$", "", text)            # an old label's note: "0.05 (default)" -> "0.05"
        return [t.replace("{}", text) for t in self.tokens.split()] if text else []


@dataclass(frozen=True)
class FamilyDescription:
    # identity
    key: str                          # family key (the workbench vocabulary), e.g. "qwen_image21"
    arch_id: str                      # architecture id used in cache filenames, e.g. "qwenimage21"
    display_name: str                 # "Qwen Image 2.1"
    gui_label: str                    # Base Model selector entry
    lora_name_suffix: str
    aliases: tuple = ()
    experimental: bool = True
    # registered (the shared cache / trainer find it) but not offered in the GUI yet - a family mid-port
    hidden: bool = False

    # model files (Preferences rows, in display order)
    model_files: tuple = ()
    text_encoder_label: str = ""
    vae_label: str = ""
    # the name its workbench files (Repair Studio presets folder, metrics table) were kept under before the family
    # moved to the driver system ("klein"); "" = its key
    shares_prefs_with: str = ""
    # the family's Preferences section: its title ("" = "<display name> model paths"), the paragraph under the title
    # ("" = a generic line), a tip at its foot (e.g. filling the paths by hand from a ComfyUI install), the line under
    # the download button ("" = generated) and the label of the button's optional tick ("" = no tick: every file
    # fetches)
    # a line under the Training tab's Base Model picker while the family is selected (e.g. what its previews are)
    model_note: str = ""
    # a grey note beside Load Preset while that preset is the selection: (preset name, note) pairs
    preset_notes: tuple = ()
    # an edit family whose workbench reference has a strength (the latents scaled by it), and whose travel renders can
    # chain - each frame also edits the previous frame's clean latent (LoRA Royale's sequential reference); the
    # original Klein engine's _build_ref_tokens. Shows the Strength boxes and the Sequential reference ticks
    reference_strength: bool = False
    # the Extract tab's presets: (name, ((block id, multiplier), ...)) - the blocks kept and the scale each is
    # extracted at; an empty block list = every block. The tab adds Custom (a block pick). () = all blocks only
    extract_presets: tuple = ()
    # the Samples tab's wording for the family, as (place, text) pairs over the generic text - places: "banner",
    # "advanced", "flow", "neg", "cfg", "steps", and "sampler" (added after the Steps line: the sampler and schedule
    # previews use, for a family sensitive to them); samples_cfg_free: previews render without CFG on a fixed schedule,
    # so the Advanced card (flow shift, negative, CFG) and the sample-model / reference rows hide
    samples_text: tuple = ()
    samples_cfg_free: bool = False
    prefs_title: str = ""
    prefs_intro: str = ""
    prefs_note: str = ""
    fetch_note: str = ""
    fetch_optional_label: str = ""
    # the "new model files" popup's opening line (the files themselves: ModelFile.announce)
    announce_intro: str = ""

    # what one training item may be: "photo", "clip" (frames + optional sound), "voice" (sound only)
    media: tuple = ("photo",)
    # Multi Concept: extra subject folders, each its own dataset block (H3: reference distillation pairs each image
    # only with others of its own subject - the reference rotation runs per block). multi_concept_defaults: the family
    # options a click on the box sets ((option key, value) pairs, still editable after); multi_concept_hint: a line
    # added to the box's hint
    multi_concept: bool = False
    multi_concept_defaults: tuple = ()
    multi_concept_hint: str = ""
    clip_spec: Optional[ClipSpec] = None

    # latent rules
    latent_channels: int = 16
    spatial_factor: int = 8
    bucket_step: int = 64             # training buckets snap to this many pixels
    image_channels: int = 3           # 4 = RGBA
    native_megapixels: float = 1.0

    # block layout (Repair Studio / block targeting)
    n_blocks: int = 0
    block_prefix: str = ""            # "transformer_blocks"
    block_note: str = ""

    # LoRA
    lora: Optional[LoRAFormat] = None

    # the family's driver ("module.path:ClassName", a FamilyDriver); empty = not built yet, so the family stays
    # hidden from training. Caching and training run through the generic entry points below for every family.
    driver: str = ""
    modelspec_arch: str = ""          # SAI modelspec.architecture, e.g. "Qwen-Image-2.1"
    training_adapter: str = ""        # pref key of the family's frozen training adapter ("" = none)
    training_adapter_note: str = ""   # one line for the Training tab under the adapter toggle
    ema_default: str = ""             # default EMA decay for the Training tab ("0.98", "Off"); "" = no EMA control
    ema_short_run: bool = False       # the EMA dropdown also offers "Short run" (decay sized to the run - H3's)
    # a saved state with no architecture tag is this family's own (the old H3 trainer wrote the same layout and
    # parameter order): resume restores its optimizer and EMA too, not just the LoRA
    resumes_untagged_states: bool = False
    implementation: str = ""          # SAI modelspec.implementation (reference repo URL)
    precisions: tuple = ("bf16",)     # base precisions offered for training: any of "bf16", "int8", "nf4"
    # what Auto may choose, in order ((): every offered precision, most precise first). Krea 2: INT8, then NF4 - its
    # original trainer's order; bf16 stays a manual choice
    auto_precisions: tuple = ()
    # the Training tab's names for the precision choices where the shared ones mislead (Klein: "bf16" loads the file
    # as it is, which for BFL's fp8 file trains in fp8), and the hint under the dropdown ("" = the shared hint)
    precision_labels: dict = field(default_factory=dict)
    precision_hint: str = ""
    # the driver can torch.compile its blocks (FamilyDriver.compile_blocks); the Training tab's Compile Blocks
    # control shows and the launch sends --compile_blocks
    compiles: bool = False
    # the generic Auto compile rule (FamilyDriver.compile_plan) for a family whose driver does not bring its own:
    # {base precision: steps after which compile has paid back its warm-up}, measured; a precision missing here is
    # not compiled by Auto (On still compiles). compile_boundary: where the gradient checkpoint sits ("inside" the
    # compiled graph - faster, more memory - or "outside" - eager-level memory); compile_fullgraph: refuse graph
    # breaks rather than degrade quietly.
    compile_payback_steps: dict = field(default_factory=dict)
    compile_boundary: str = "inside"
    compile_fullgraph: bool = True
    # measured peak GB when compiled: {precision: {"inside" | "outside": ((megapixels, GB), ...)}}. With it the compile
    # plan checks free VRAM: the family's boundary if it fits, else the checkpoint outside the graph, else uncompiled.
    # {} = no memory check (the compiled peak is no bigger than eager's)
    compile_memory: dict = field(default_factory=dict)
    # the Training tab's hint under Compile Blocks: this family's measured figures
    compile_hint: str = ""
    # previews can take the Samples tab's reference image through the text encoder's vision path ('prompt from a
    # picture', Krea 2) - conditioning only, not an edit reference (edit_training)
    preview_image: bool = False
    # how the workbench samples its preview checkpoint (the model file with role "preview_dit"), which it uses by
    # default for fast previews when the file is set; None = the family has none
    preview_checkpoint_sampling: Optional[SamplingSettings] = None
    # the optimizer state waits on CPU while the previews render (H3: ~2.5 GB of dead weight for a no-grad render)
    preview_park_optimizer: bool = False
    # a full fine-tune of the base model is offered (the driver's ft_spec returns its FTSpec, families/ft.py): the
    # Training tab shows the fine-tune card and the launch sends --finetune. Optional - most families never need it
    finetune: bool = False
    ft_learning_rate: float = 1e-5    # the learning rate choosing Fine-tune sets (H3's tested rate is 3e-5)
    # the Training tab's large-dataset hint: (rate, steps per epoch) - a standard LoRA whose learning rate (or
    # adaptive Min LR) is at least `rate`, on more than `steps` steps an epoch, gets a tip suggesting a cooler rate,
    # so the best point doesn't fall between two saved epochs. None turns the hint off for the family
    lr_hint: tuple = (2e-4, 125)
    # training previews may render on the preview checkpoint (the preview_dit file, sampled with
    # preview_checkpoint_sampling) instead of the training model - Klein's Distilled previews. The driver brings the
    # memory handoff (park_for_preview / load_preview_checkpoint / unpark_after_preview)
    train_preview_checkpoint: bool = False
    # measured training memory for the Auto plan: {precision: (peak GB with no block swap, GB saved per swapped
    # block)}; the peak may instead be ((megapixels, GB), ...) points, interpolated for the run's resolution.
    # {} = Auto just takes the first precision
    train_memory: dict = field(default_factory=dict)
    optimizers: tuple = ("adamw8bit", "adamw")
    # optimizer settings a family's own trainer applied (only when Optimizer Args doesn't set them): an Adam-family
    # weight decay, and the 8-bit Adam eps floor of 1e-6 (training/optimizers.create_optimizer eps_floor_8bit - H3's
    # fix for 8-bit second moments underflowing on its most structured tensors)
    optimizer_weight_decay: Optional[float] = None
    optimizer_eps_floor_8bit: bool = False
    # the trainable adapter's weights: "fp32" (the layer's) or "bf16" (MiniMax H3's old trainer)
    trainable_dtype: str = "fp32"
    # Automagic v3 keeps one learning rate per parameter group: (group name, (substrings of the dotted module
    # name, ...)) splits the LoRA so each family of modules finds its own rate; the first match wins, the rest form
    # "other". () = one group
    optimizer_families: tuple = ()
    # Automagic v3's sign window for this family when Optimizer Args doesn't set polarity_history (0 = its default)
    automagic_sign_window: int = 0
    network_types: tuple = ("lora",)
    # Adaptive LR also treats a grad-clip ratio over 50% of an epoch's steps as a stability signal (Klein's rule,
    # training/adaptive_lr.AdaptiveLR clip_signal)
    adaptive_lr_clip_signal: bool = False
    adaptive_lr: bool = True          # the Adaptive LR control (off: hidden, never sent - MiniMax H3)
    loss_watch: bool = True           # the per-image loss watch toggles (off: hidden, never sent - MiniMax H3)
    network_hint: str = ""            # the line under Network Type in place of the LoRA / LoKR trade
    ema_hint: str = ""                # the line under Weight averaging (EMA) in place of the shared one
    ema_section: str = ""             # "other": the EMA row sits in Other Options (MiniMax H3's place)
    precision_label: str = ""         # the Base precision row's label ("Base Precision")
    precision_after_states: bool = False   # the Base precision row below Save State / Keep Last (H3's place)
    # the Samples tab's "Turbo preview: N steps at M% strength" row (MiniMax H3) instead of the Turbo strength box:
    # the Steps box is the plain-model count, the row the turbo's
    samples_turbo_pace: bool = False
    edit_training: bool = False       # Edit LoRA from before/after pairs (the driver's supports_references)
    edit_note: str = ""               # the Edit LoRA section's "What you need" line: pair count and photo size
    slider_training: bool = False     # Slider LoRAs (strength is a dial between two looks); the driver needs
    #                                   training_loss(diff_ref=, diff_weight=) and noise_latents / predict
    slider_guidance: float = 2.0      # a prompt slider's default push strength (the Training tab box when empty)
    slider_ultra_blocks: tuple = ()   # block ids (driver.block_map) an "Ultra mode" slider trains, () = no Ultra mode:
    #                                   the composition blocks only, so the slider holds up at far higher strengths

    # sampling
    sampling: tuple = ()              # SamplingSettings without any speed LoRA (first = default)
    speed_loras: tuple = ()           # SpeedLoRA entries
    preview_steps: int = 20
    preview_cfg_note: str = ""         # the Samples tab's line under CFG Scale ("" = a plain default line)
    # the Samples tab's default negative prompt for the family (each family keeps the user's own edit); None = its
    # previews take no negative, so the box greys out
    preview_negative: Optional[str] = None
    # Repair Studio / Explorer / Royale renders take comfy-kitchen's INT8 attention (fizgig/modules/int8_attention.py:
    # the family's attention calls attend() first). Inference only; the Profiler keeps exact attention
    int8_attention: bool = False
    # Repair Studio's Turbo Preview tick (families/act_cache.py): a slider change replays the unchanged blocks on
    # step 1 - the same picture as a full render
    activation_cache: bool = False
    workbench_follows_samples: bool = False   # Repair Studio / Explorer / Royale previews take the Samples tab's steps,
    #                                   CFG, negative prompt and turbo strength (else the family's fixed preview recipe)
    preview_cfg: float = 1.0
    preview_width: int = 1024
    preview_height: int = 1024
    repair_size: int = 768            # Repair Studio's starting preview size (its Res dropdown) for the family
    preview_speed_lora: str = ""      # name of the SpeedLoRA previews use when its file is set in Preferences
    preview_speed_steps: int = 0      # preview steps with it (0 = the SpeedLoRA's own)
    preview_speed_strength: Optional[float] = None   # preview strength for it (None = the SpeedLoRA's own; 0 = off
    # by default: previews render without it until the Samples tab's Turbo strength is raised)

    # built-in Training-tab presets: ((name, {GUI setting key: value}), ...); the first is applied on a first visit
    presets: tuple = ()

    # small files the text encoder / captioner load by repo name: ((repo, (allow_patterns...)), ...); the model
    # downloader fetches them with the helper models so first use works offline
    helper_files: tuple = ()

    # workbench tools that support this family ("repair", ...); the generic WorkbenchEngine drives them all
    workbench: tuple = ()
    # the workbench engine class ("module:Class") when the generic engines do not fit (a family with clips gets the
    # generic video engine, families/video_workbench.py). H3 brings its own for keyframes, references and RefMod on
    # top of the shared clip contract. Built as Class(description).
    workbench_engine: str = ""
    # a video family's clip render regimes in the workbench: ((name, steps, speed LoRA strength), ...) - H3's "dial"
    # (the fast loop) and "confirm" (the render that matches training previews). () = one regime, the speed LoRA's
    # own steps and strength (or the default sampling without one)
    clip_regimes: tuple = ()
    # Repair Studio built-ins beyond Reset All: (name, ((block id, strength), ...)) - every other block at 1.0
    repair_presets: tuple = ()
    # what each block carries, measured with the Profiler: ((block id, category), ...), category one of
    # "identity" / "look" / "style_ident_overlap" (Repair Studio's colours). Unlisted blocks stay uncoloured.
    block_categories: tuple = ()
    # Repair Studio's category controls over block_categories - the five master sliders, the donor category toggles
    # and the Identity / Style+Composition / Details only presets (Klein's 5-bucket map: style_composition,
    # style_ident_overlap, identity, ident_details_overlap, details). False = per-block sliders only
    category_masters: bool = False
    # Fast Identity Mode: the block ids a standard LoRA trains alone when it is on (the measured identity blocks).
    # Everything before the first of them runs forward only, so it is faster. () = no Fast Identity Mode.
    identity_blocks: tuple = ()
    # Model Area to Train: ((name, (block ids...), (min_t, max_t) or None), ...) - () blocks = every block, a window
    # (0-1, 0 = clean) fills the Timestep Range. The first entry is the default; the Training tab adds "Custom" (pick
    # blocks) and shows the Timestep Range (0-1000) for a family that has areas. () = neither control
    train_areas: tuple = ()

    # things a user or a later session must know, with sources
    notes: tuple = ()
    options: tuple = ()               # FamilyOption controls (the family's own settings, as launch tokens)
    # a saved preset / Last Train written by the family's old entry: {standard-layer key: old key} filled in when the
    # preset lacks the new key (H3's MINIMAX_EMA -> FAMILY_EMA, its fine-tune card's keys, ...)
    settings_aliases: dict = field(default_factory=dict)

    # ---- derived ------------------------------------------------------------------------------
    # generic entry points shared by every described family (the standard layer)
    train_script = "src/fizgig/families/train.py"
    cache_script = "src/fizgig/families/cache.py"

    @property
    def training_ready(self) -> bool:
        return bool(self.driver)

    @property
    def reference_kind(self) -> str:
        """How a reference picture reaches previews and the workbench: "vision" (the text encoder sees it,
        preview_image), "edit" (the prompt edits it, edit_training) or "" (none)."""
        return "vision" if self.preview_image else ("edit" if self.edit_training else "")

    def preview_checkpoint(self):
        """(ModelFile, SamplingSettings) of the workbench's preview checkpoint, or None."""
        f = next((m for m in self.model_files if m.role == "preview_dit"), None)
        return (f, self.preview_checkpoint_sampling) if f is not None and self.preview_checkpoint_sampling else None

    def load_driver(self):
        """Instantiate the family's FamilyDriver (imported lazily: the description stays importable without torch)."""
        import importlib
        mod, _, cls = self.driver.partition(":")
        drv = getattr(importlib.import_module(mod), cls)()
        drv.description = self
        return drv

    def workbench_engine_class(self):
        """The class make_workbench_engine builds - read for its capabilities (supports_keyframes, ...) before any
        engine exists."""
        import importlib
        if self.workbench_engine:
            mod, _, cls = self.workbench_engine.partition(":")
            return getattr(importlib.import_module(mod), cls)
        if "clip" in self.media:
            from fizgig.families.video_workbench import VideoWorkbenchEngine
            return VideoWorkbenchEngine
        from fizgig.families.workbench import WorkbenchEngine
        return WorkbenchEngine

    def make_workbench_engine(self):
        """The family's workbench engine: its own class (workbench_engine), else the generic video engine for a family
        with clips, else the generic WorkbenchEngine."""
        return self.workbench_engine_class()(self)

    @property
    def video_workbench(self) -> bool:
        """The workbench tabs' clip features apply (a family whose media include clips)."""
        return "clip" in self.media

    def lora_prefix(self, block: int, module: str) -> str:
        """Key stem of one wrapped module, e.g. 'transformer.transformer_blocks.3.attn.to_q'."""
        return self.lora.key_template.split(".{ab}")[0].format(block=block, module=module)

    @property
    def pref_keys(self) -> tuple:
        return tuple(f.pref_key for f in self.model_files)

    @property
    def required_pref_keys(self) -> tuple:
        return tuple(f.pref_key for f in self.model_files if f.required)

    def pref_for(self, role: str) -> str:
        """Pref key of the model file with this role ("" if the family has none)."""
        return next((f.pref_key for f in self.model_files if f.role == role), "")

    def model_path(self, role: str, lookup) -> str:
        """The path for a role, given lookup(pref_key) -> the user's setting: the role's own file, or - for a part
        that ships inside another file (ModelFile.inside) and is left empty - that file's."""
        f = next((m for m in self.model_files if m.role == role), None)
        if f is None:
            return ""
        p = str(lookup(f.pref_key) or "").strip()
        if not p and f.inside:
            p = str(lookup(f.inside) or "").strip()
        return p

    def block_ids(self) -> list:
        return [f"{self.block_prefix}_{i}" for i in range(self.n_blocks)]

    def preview_speed(self):
        """The SpeedLoRA used for in-training previews, or None."""
        return next((sl for sl in self.speed_loras if sl.name == self.preview_speed_lora), None)

    def preview_speed_defaults(self):
        """(steps, strength) previews use with the preview SpeedLoRA, or None without one."""
        sp = self.preview_speed()
        if sp is None:
            return None
        return (self.preview_speed_steps or sp.settings.steps,
                sp.strength if self.preview_speed_strength is None else self.preview_speed_strength)

    def default_sampling(self) -> Optional[SamplingSettings]:
        return self.sampling[0] if self.sampling else None

    def architecture_entry(self) -> dict:
        """The ARCHITECTURES-shaped dict the GUI's existing config readers expect, so start_training,
        the Samples tab and validation can read a description family without KeyErrors. Klein-shaped
        keys that do not apply are filled with the neutral values the Krea 2 entry uses."""
        s = self.default_sampling()
        return {
            "family_key": self.key,
            "is_description_family": True,
            "train_script": self.train_script,
            "cache_latents_script": self.cache_script,
            "cache_text_script": self.cache_script,
            "use_fizgig_venv": True,
            "blocks_swap_max": max(0, self.n_blocks - 2),
            "fp8_text_encoder_flag": None,
            "uses_clip": False,
            "uses_t5": False,
            "uses_text_encoder": True,
            "uses_model_type": False,
            "uses_model_version": False,
            "model_version": self.arch_id,
            "vae_label": self.vae_label,
            "text_encoder_label": self.text_encoder_label,
            "is_distilled": False,
            "supports_samples": True,
            # a CFG-free family (samples_cfg_free) greys Negative Prompt and CFG Scale, as the distilled flags do; a
            # family with no preview_negative greys the negative
            "sample_is_distilled": bool(self.samples_cfg_free or self.preview_negative is None),
            "sample_negative_default": self.preview_negative,
            "sample_cfg_fixed": bool(self.samples_cfg_free),
            "sample_cfg_default": self.preview_cfg,
            "sample_flow_shift_default": None,
            "sample_steps_default": self.preview_steps if self.preview_steps else (s.steps if s else 20),
            "sample_width_default": self.preview_width,
            "sample_height_default": self.preview_height,
            "lora_name_suffix": self.lora_name_suffix,
        }

    def validate(self) -> list:
        """Internal consistency problems (empty list = fine). Cheap; run by the registry and tests."""
        problems = []
        if not (self.key and self.arch_id and self.display_name and self.gui_label):
            problems.append("identity fields must all be set")
        keys = [f.pref_key for f in self.model_files]
        if len(keys) != len(set(keys)):
            problems.append("duplicate pref keys")
        if self.n_blocks <= 0 or not self.block_prefix:
            problems.append("block layout missing")
        if self.lora is None:
            problems.append("LoRA format missing")
        if not self.sampling:
            problems.append("no sampling settings")
        if self.bucket_step % self.spatial_factor:
            problems.append("bucket_step must be a multiple of spatial_factor")
        for sl in self.speed_loras:
            if not (sl.repo and sl.file and sl.source):
                problems.append(f"speed LoRA {sl.name!r} is missing repo/file/source")
        if self.driver and ":" not in self.driver:
            problems.append("driver must be 'module.path:ClassName'")
        if self.driver and not all(self.pref_for(r) for r in ("dit", "vae", "text_encoder")):
            problems.append("a trainable family needs model files with roles dit, vae and text_encoder")
        if self.training_adapter and self.pref_for("training_adapter") != self.training_adapter:
            problems.append("training_adapter must name the model file whose role is training_adapter")
        if self.driver and not (self.modelspec_arch and self.implementation):
            problems.append("a trainable family needs modelspec_arch and implementation for LoRA metadata")
        return problems
