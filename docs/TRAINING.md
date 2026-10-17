# Training features and the workbench

[← Back to the README](../README.md)

Everything on this page works on every model unless it says otherwise: [Klein 9B](KLEIN.md), [Flux 3 Image Klein](FLUX3.md), [Krea 2](KREA2.md), [MiniMax H3](MINIMAX_H3.md) and [Qwen Image 2.1](QWEN_IMAGE.md).

## The sample gallery is an instrument

- **Live likeness scoring** — pick 3 dataset photos and every sample gets a colour-coded likeness badge (ArcFace for people, DINOv2 for furry / anime: [face backends](FACE_BACKENDS.md); CPU — zero training-speed cost), with a per-epoch trend chart and best-epoch highlight, live while the run goes.
- **Training Run Visualiser** — scrub the run epoch by epoch in the browser, Royale-style, with share-ready WebM/PNG export.
- **Every epoch checkpoint carries its own preview as the thumbnail**, so file browsers and LoRA managers show what each epoch looks like.
- **Change the previews of the run going now** (a tick on the Samples tab) changes the preview prompts, seed and size mid-run, no restart; **Sample now** and **Save now** on the run card render or save at the next step. The top bar carries VRAM/RAM gauges.

## Dataset prep

- **AI captioning with Krea 2's Qwen3-VL-4B** — every model's captions, on the Captions tab and in auto-recaption, come from it. It writes viewpoint-aware training captions in five editable preset styles (including **Style**, which describes everything *except* the look so your trigger word binds to it). Every preset's instruction is editable in plain English and persists. Florence-2 is the zero-setup option. **Bilingual captions** (English + Chinese via Helsinki-NLP) act as text-level augmentation — measurably better skin detail on Klein at identical loss.
- **Image Prep** — batch resize, PNG conversion, face crops (InsightFace with gender targeting for people, or a YOLO face model such as Fdetailer for furry / anime characters: [face backends](FACE_BACKENDS.md)). Pairing a tight crop with a full shot adds a lot to a character dataset.
- **Look Consistency Filter** — pick the 3 images that best nail the look and every image is scored against them (ArcFace, or DINOv2 for furry / anime faces). Worst matches surface first; mark drifters or let Auto-Suggest flag the outliers, then move them out in one click — nothing is deleted, and the scores feed the trainer's [look-outlier warm-up](KREA2.md#the-trainer-curates-your-dataset-while-it-trains-experimental).

## Subjects: more than one person, and keeping them to their trigger

The Training tab's **Subjects** card says who the LoRA is of and how it behaves with other people. 1 subject with Off is ordinary LoRA training. Every other setting is Fizgig's own techniques combined with DOP (Differential Output Preservation, from @ostris's AI-Toolkit).

- **Who is this LoRA of?** 1 to 4 subjects, each a folder of their photos with their own trigger word in every caption. Subject 1 is the Start tab's folder.
- **Your subject and other people:**
  - **Off - a normal LoRA:** your subject's face can leak onto other people, even in prompts that don't use its trigger word.
  - **Leave the rest of the model alone:** ask for "a man at a party" without the trigger word and you get a stranger, just as the model made before you trained.
  - **Fizgig LoRA Isolation:** your subjects can stand together in one picture, or next to someone the model already knows, and every face stays right. Best for people; about 1.5x the training time; its 21 MB of strangers come with any model's download in Preferences. It joins each trigger to what the subject is (`zwxem, a woman` trains as `zwxemwoman`, your caption files untouched) and adds 16 side-by-side pictures per subject next to fixed strangers. **Use my captions exactly as written** turns the joining off for anyone who writes the joined word themselves.
- At the bottom of the card, each subject shows one of its own captions and exactly what training makes of it, and it updates as you change the settings above.
- Two subjects of the same kind (two women) need about 1.6x the epochs; the Training tab says so.
- Up to 4 people work well, 2 or 3 of them in one picture. With a person the model already knows well in the picture too, 2 subjects is reliable and 3 is hit and miss: see [How many subjects](MULTI_PERSON.md#how-many-subjects).

How to caption for it, choose trigger words and prompt the finished LoRA: [People in group pictures](MULTI_PERSON.md).

## Output folders

The LoRA output folder lives on the Training tab's Output section and is remembered per model: switch models and each gets its own folder back. The default is `output_loras` inside Fizgig.

## Compatibility

Loads kohya, PEFT, OneTrainer (OMI + legacy), AI-Toolkit, and LyCORIS (LoKR / LoHa) — auto-converted, and LoKR/LoHa run natively everywhere: Repair Studio, Profiler, Extract, Context LoRA. Repair Studio saves LoKR as LoKR, losslessly. Output is `.safetensors` that drops straight into ComfyUI.

## The workbench

Each tool works on a trained run's output **or any LoRA you've downloaded** — and they hand off to each other (profile → repair → explore → compare, one closed loop). MiniMax H3 previews render a short clip, judged by its middle frame — the model's native regime.

### Repair Studio

A live slider per transformer block (32 on Klein, up to 50 + the token refiners on MiniMax H3) with a side-by-side preview that updates as you drag. **Turbo Preview** caches per-block activations so late-block edits redraw up to 97% faster; the baked save is always exact. Blend blocks from a second **donor** LoRA, balance the pair per block, condition previews on a reference photo, and save a `.safetensors` that works in ComfyUI at strength 1.0. On MiniMax H3 the previews are **clips with sound**, played side by side in the app, with a per-block library and first/last-frame conditioning — see [Repair Studio on H3](MINIMAX_H3.md#repair-studio-on-h3). Krea 2 has its own **✨Text fusion** presets — see [KREA2.md](KREA2.md#help-map-krea-2s-blocks).

### Exploring variations

Repair Studio's **🎲 I'm feeling lucky** rolls a new strength for each of the LoRA's blocks between your Min and Max (locked blocks keep theirs; Hold keeps each part of the model in its own safe range), renders it, and leaves the sliders where it landed: keep a roll you like, roll again or Undo, lock the blocks you like and roll the rest, or nudge single blocks from there. It does what the old desktop app's LoRA the Explorer did, on the same page as the fixing and saving. A **load strength** box sets the strength the LoRA is meant to be used at.

### LoRA Royale

Point it at a training run and it renders **every epoch on one fixed seed**, with a crossfade slider — drag until it looks best and stop. An optional **likeness score** (the Image Prep tab's look match, CPU) rates each epoch against a training photo and jumps you to the best. Then make it shareable: epoch-morph clips, seed / prompt / strength **travels**, a **comparison sheet** (with/without-LoRA grid, same seed per row), all exportable as looping MP4/GIF with an optional deflicker pass. Works on any folder of LoRAs, or a single file.

### Profiler

How much rank a LoRA really uses and where its weights are, then renders with each block group switched off and scores likeness and bleed — which blocks carry the subject and which leak it. The result is a colour-coded HTML report. Repair Studio reads its sidecar automatically and shows the findings inline when you load the same LoRA. Klein's block map is in [KLEIN.md](KLEIN.md#block-map).

### Extract

Distil any Klein, Flux 3 Image Klein, Krea 2, MiniMax H3 or Qwen Image 2.1 LoRA to a lower rank — it runs weight-only SVD with no models loaded, and Klein's presets keep just the Identity, Style+Composition or Details blocks (or any blocks you pick). PEFT and LyCORIS sources supported.
