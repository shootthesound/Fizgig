# Training features and the workbench

[← Back to the README](../README.md)

Everything on this page works on every model unless it says otherwise: [Klein 9B](KLEIN.md), [Flux 3 Image Klein](FLUX3.md), [Krea 2](KREA2.md), [MiniMax H3](MINIMAX_H3.md) and [Qwen Image 2.1](QWEN_IMAGE.md).

## The sample gallery is an instrument

- **Live likeness scoring** — pick 3 dataset photos and every sample gets a colour-coded likeness badge (ArcFace for people, DINOv2 for furry / anime: [face backends](FACE_BACKENDS.md); CPU — zero training-speed cost), with a per-epoch trend chart and best-epoch highlight, live while the run goes.
- **Training Run Visualiser** — scrub the run epoch by epoch in the browser, Royale-style, with share-ready WebM/PNG export.
- **Every epoch checkpoint carries its own preview as the thumbnail**, so file browsers and LoRA managers show what each epoch looks like.
- **How often to preview** (Samples tab): **Every** N epochs, and, switched on, **then from epoch** Y, **every** Z epochs (whole multiples). Switching it on fills in every 5, then from epoch 10 every 1: few previews early, every epoch once the run gets close. Off, previews run every N epochs throughout.
- **How often to save** (Training tab) works the same way for checkpoints: **Every** N epochs, and, switched on, **then from epoch** Y, **every** Z epochs. The last epoch is always saved. A fine-tune saves at whole rotations, so it doesn't use it.
- **Change the previews of the run going now** (a tick on the Samples tab) changes the preview prompts, seed, size, steps, CFG, Turbo LoRA strength and how often they run mid-run from the next preview, no restart (a Turbo strength works on a run started at 0 too, as long as the Turbo LoRA file is set in Preferences); **Resume** also takes the Samples tab's preview settings as they are now, the preview-checkpoint tick (SFT, Distilled) included, block swap and the learning rate (a new Learning Rate, or adaptive LR Min / Max, is used from there on and the console says so); **Sample now** and **Save now** on the run card render or save at the next step. The top bar carries VRAM/RAM gauges.
- **Prompt sets** (under the Samples tab's prompts): **Save as set…** keeps the prompts under a name, and **Load** puts a saved set in place of the prompts there. Sets are shared by every model, so a model you train for the first time can start with your usual prompts; with **Change the previews of the run going now** ticked, a loaded set goes to the run training now from its next preview.

## Dataset prep

- **AI captioning with Krea 2's Qwen3-VL-4B** — every model's captions, on the Captions tab and in auto-recaption, come from it. It writes training captions from editable presets at four lengths, each in a version for a person or character and one for an object or product, plus **Style**, which describes everything *except* the look so your trigger word binds to it. Every preset's instruction is editable in plain English and persists. Auto-recaption uses them too: the Training tab's **First try uses** and **Second try uses** pick a task each (Training caption, then Exhaustive detail, by default). Both defaults are the person versions, so for an object pick the object or product versions, for a style Style, or Custom with your own instruction. Florence-2 is the zero-setup option. **Bilingual captions** (English + Chinese via Helsinki-NLP) act as text-level augmentation — measurably better skin detail on Klein at identical loss.
- **Image Prep** — batch resize, PNG conversion, face crops (InsightFace with gender targeting for people, or a YOLO face model such as Fdetailer for furry / anime characters: [face backends](FACE_BACKENDS.md)). Pairing a tight crop with a full shot adds a lot to a character dataset.
- **Masks for masked training** — Image Prep makes a mask for every picture in a folder (the person, just the face and hair, the body without the face, and more), so training learns the part that matters, not the room around it. See [Masked training](MASKED_TRAINING.md).
- **Look Consistency Filter** — pick the 3 images that best nail the look and every image is scored against them (ArcFace, or DINOv2 for furry / anime faces). Worst matches surface first; mark drifters or let Auto-Suggest flag the outliers, then move them out in one click — nothing is deleted, and the scores feed the trainer's [look-outlier warm-up](KREA2.md#the-trainer-curates-your-dataset-while-it-trains-experimental).
- **Sub-folders (optional)** — tick **Also train on its sub-folders** on the Training tab and a big dataset can stay organised in folders (outfits, forms, sources), each trained with its own captions and cache. Off by default. See [Sub-folders](#sub-folders) below.

## Sub-folders

A training folder is read at its top level only. Tick **Also train on its sub-folders** on the Training tab (off by default) and the pictures in its sub-folders train too, so a big dataset can stay organised in folders: outfits, forms, sources, or the thousands of pictures of a fine-tune. The folder you choose on the Start tab stays the training folder; the tick only adds what is inside it.

**What it trains.** Every sub-folder holding pictures (or clips or songs) is found, at any depth. Each trains as its own set beside the training folder, with its own captions next to its pictures and its own cache, so two folders can hold pictures with the same name. Every picture trains once an epoch, wherever it is: a folder with more pictures carries more weight, and nothing is evened out between folders. The training folder itself can hold no pictures at all, with everything in its sub-folders.

**What it leaves out.** Fizgig's own folders, matched in any case (Image Prep's `originals`, and `originals_2`, `originals_3`… from later runs; the Look Consistency Filter's `excluded_by_look`; and `removed`, where Remove puts pictures), hidden folders (a name starting with `.`), the LoRA output folder and the cache folder when they sit inside, and a fine-tune's regularisation folder (below), with everything under them. The run's log names the Fizgig folders it left out.

**The line under the tick** says what it will do before you press Start:
- Unticked, when there are sub-folders: how many there are and how many more pictures they hold, which aren't trained on.
- Ticked: how many folders and pictures will train, and first the folders whose pictures aren't captioned yet (the training folder's own too), with how many pictures each. **Show all folders** lists every folder with its count.
- Start refuses a sub-folder whose pictures have no captions and names it, as it does for the training folder.

**Captions and Image Prep** work on one folder at a time, and their lines say so when the training folder has sub-folders. To caption a sub-folder, make its masks, resize it or run the Look Consistency Filter on it, choose it as the training folder on the Start tab, then choose the top folder again to train. Each sub-folder's trigger word is the one you saved for it on the Captions tab, else the word its captions start with. With masked training on, each sub-folder's pictures use the masks in their own folder.

**While it trains** (the Krea 2 loss watch, [details](KREA2.md#the-trainer-curates-your-dataset-while-it-trains-experimental)), every picture is kept with its own folder:
- **Per-image loss** (on the run card while it trains, and on the run in Runs & outputs) names each picture with its folder (`outfits/red/03`), and a caption you edit there is written into that picture's own folder.
- **Auto-recaption** writes the new caption into the picture's own folder, with that folder's trigger word.
- **Exclusions** are saved in each folder's own `fizgig_excluded.json`, so they travel with its pictures.
- The **look-outlier warm-up** reads each folder's own Look Filter scores (`fizgig_look_scores.json`, from a scan of that folder). Pictures moved into a sub-folder's `excluded_by_look` are left out.
- Pause and Resume carry all of it on, and **Load settings from last train** brings the tick back.

**Fine-tunes** train every folder in the tree, which is the main use: a dataset of thousands of pictures sorted into folders. If the folder you picked for **Regularisation images** sits inside your training folder, its photos are not trained as your own pictures: they are only used as regularisation images, as they would be from any other folder.

**On a pod or another computer** the tick decides what goes:
- **Upload a folder** on the Start tab keeps the sub-folders as they are when it is ticked. Unticked, the files of the sub-folders all go into the one new folder.
- **Send to this pod** (the Fizgig window's **From this computer** card) sends the sub-folders too when it is ticked; each tile says whether its sub-folders go ("sent too" or "not sent"). Sending again sends only what isn't there yet.
- **Move a run** takes the sub-folders with the run, and captions edited over there come back into each folder. See [Move a run](MOVE_RUN.md).

**Not with** Subjects 2-4, DOP or Fizgig LoRA Isolation, which keep one folder per subject, nor Edit or slider training, which pair their files: the tick is hidden in those modes. To train several people, use Subjects; to train one subject's outfits or forms, use sub-folders or one folder with the outfit named in each caption.

For the command line, see [CLI](CLI.md): each sub-folder is a `[[datasets]]` block with an `item_prefix`.

## Subjects: more than one person, and keeping them to their trigger

The Training tab's **Subjects** card says who the LoRA is of and how it behaves with other people. 1 subject with Off is ordinary LoRA training. Every other setting is Fizgig's own techniques combined with DOP (Differential Output Preservation, from @ostris's AI-Toolkit).

- **Who is this LoRA of?** 1 to 4 subjects, each a folder of their photos with their own trigger word in every caption. Subject 1 is the Start tab's folder.
- **Your subject and other people:**
  - **Off - a normal LoRA:** your subject's face can leak onto other people, even in prompts that don't use its trigger word.
  - **Leave the rest of the model alone:** ask for "a man at a party" without the trigger word and you get a stranger, just as the model made before you trained. With one person, use it when your subject must stay out of prompts without the trigger word; otherwise use Off. **Fizgig LoRA Isolation** needs two or more subjects: with one, it's greyed out.
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
