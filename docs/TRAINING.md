# Training features and the workbench

[← Back to the README](../README.md)

Everything on this page works on every model unless it says otherwise: [Klein 9B](KLEIN.md), [Flux 3 Image Klein](FLUX3.md), [Fibo 2.0](FIBO2.md), [Krea 2](KREA2.md), [MiniMax H3](MINIMAX_H3.md) and [Qwen Image 2.1](QWEN_IMAGE.md).

## The sample gallery is an instrument

- **Live likeness scoring** — pick 3 dataset photos and every sample gets a colour-coded likeness badge (AdaFace ViT for people, DINOv2 for furry / anime: [face backends](FACE_BACKENDS.md); CPU — zero training-speed cost), with a per-epoch trend chart and best-epoch highlight, live while the run goes. The chart also shows **your photos**: your own training photos scored against the same 3, the realistic top a preview can reach (even real photos of a person never score 100%), and the run line says whether the best epoch is within, level with or above them. The three photos are remembered with the training folder, so every run of the same person uses them without picking again, and each run keeps its own scores: switching between runs in the gallery never clears or mixes them. The 🎨 Style score works the same way: switched on for a run, it stays on for that training folder's other runs, and each run keeps its own scores.
- **Star the previews you like** — the ☆ in a preview's top corner (always shown on a phone) stars it without opening it; starred previews get a gold frame, and when the run is done the checkpoint lists (the run card, Runs & outputs) put a ★ beside each starred epoch's checkpoint, so your shortlist is waiting there. LoRA Royale renders just those epochs: **Max renders > Starred** on the crossfade, or **★ Starred** beside the comparison sheet's Epochs box for a column per starred epoch. Stars live in the samples folder with the previews.
- **Training Run Visualiser** — scrub the run epoch by epoch in the browser, Royale-style, with share-ready WebM/PNG export.
- **Every setting these previews use** — under the gallery's preview line: the model, the run's LoRA and its strength (and whether it is the EMA weights), sampler, scheduler, steps, CFG, the negative prompt, the size and the seeds, recorded by the trainer beside the previews. **Copy** puts the list on the clipboard, to match a preview in ComfyUI.
- **Every epoch checkpoint carries its own preview as the thumbnail**, so file browsers and LoRA managers show what each epoch looks like.
- **How often to preview** (Samples tab): **Every** N epochs, and, switched on, **then from epoch** Y, **every** Z epochs (whole multiples). Switching it on fills in every 5, then from epoch 10 every 1: few previews early, every epoch once the run gets close. Off, previews run every N epochs throughout.
- **How often to save** (Training tab) works the same way for checkpoints: **Every** N epochs, and, switched on, **then from epoch** Y, **every** Z epochs. The last epoch is always saved. A fine-tune saves at whole rotations, so it doesn't use it.
- **Change the previews of the run going now** (a tick on the Samples tab) changes the preview prompts, seed, size, steps, CFG, Turbo LoRA strength and how often they run mid-run from the next preview, no restart (a Turbo strength works on a run started at 0 too, as long as the Turbo LoRA file is set in Preferences); **Resume** also takes the Samples tab's preview settings as they are now, the preview-checkpoint tick (SFT, Distilled) included, block swap and the learning rate (a new Learning Rate, or adaptive LR Min / Max, is used from there on and the console says so); **Sample now** and **Save now** on the run card render or save at the next step. The top bar carries VRAM/RAM gauges.
- **Prompt sets** (under the Samples tab's prompts): **Save as set…** keeps the prompts under a name, and **Load** puts a saved set in place of the prompts there. Sets are shared by every model, so a model you train for the first time can start with your usual prompts; with **Change the previews of the run going now** ticked, a loaded set goes to the run training now from its next preview.

## Dataset prep

- **AI captioning with Krea 2's Qwen3-VL-4B** — every model's captions, on the Captions tab and in auto-recaption, come from it. It writes training captions from editable presets at four lengths, each in a version for a person or character and one for an object or product, plus **Style**, which describes everything *except* the look so your trigger word binds to it. Every preset's instruction is editable in plain English and persists. Auto-recaption uses them too: the Training tab's **First try uses** and **Second try uses** pick a task each (Training caption, then Exhaustive detail, by default). Both defaults are the person versions, so for an object pick the object or product versions, for a style Style, or Custom with your own instruction. By default the first try is given the picture's caption to correct (**First try corrects the caption you wrote, instead of starting over**, ticked): it changes the details the picture doesn't show and keeps your words and terms, which matters most for concept LoRAs; the second try still writes a new caption. Untick it for a new caption from the picture on both tries. Your caption from before auto-recaption is kept beside the pictures (`fizgig_caption_history.json`), and **Edit caption** on the image's tile in the Problem Images window offers it back with **Put it back**. Florence-2 is the zero-setup option. **Bilingual captions** (English + Chinese via Helsinki-NLP) act as text-level augmentation — measurably better skin detail on Klein at identical loss.
- **Image Prep** — batch resize, PNG conversion, face crops (InsightFace with gender targeting for people, or a YOLO face model such as Fdetailer for furry / anime characters: [face backends](FACE_BACKENDS.md)). Pairing a tight crop with a full shot adds a lot to a character dataset.
- **Masks for masked training** — Image Prep makes a mask for every picture in a folder (the person, just the face and hair, the body without the face, and more), so training learns the part that matters, not the room around it. See [Masked training](MASKED_TRAINING.md).
- **Look Consistency Filter** — pick the 3 images that best nail the look and every image is scored against them (AdaFace ViT, or DINOv2 for furry / anime faces). Worst matches surface first; mark drifters or let Auto-Suggest flag the outliers, then move them out in one click — nothing is deleted, and the scores feed the trainer's [look-outlier warm-up](KREA2.md#the-trainer-curates-your-dataset-while-it-trains-experimental).
- **Sub-folders (optional)** — tick **Also train on its sub-folders** on the Training tab and a big dataset can stay organised in folders (outfits, forms, sources), each trained with its own captions and cache. Off by default. See [Sub-folders](#sub-folders) below.
- **Several captions per picture (optional)** — a caption file can hold several captions, one a line: a line of tags and a sentence, or captions from different captioners. See [Several captions per picture](#several-captions-per-picture) below.

## Element boost: one part of each picture learns faster

**Element boost (experimental)**, in Training Parameters, for MOSS-Image and MiniMax H3 Image. Early in the run one part of each picture counts much more than the rest, so it's learned sooner: by default the face and hair, which brings likeness in faster, most of all when faces are small in the frame. It runs three boosted stretches then one normal, for the first 12 epochs or 600 steps, whichever comes first; after that the run trains as normal, so the body and background catch up. It works with or without masks (a mask trims it), and a picture where the part isn't found trains as normal. It's on in MOSS-Image's Standard preset; on MiniMax H3 Image it's off in every preset, so tick it to try it.

**It needs Target Megapixels 0.75 or more** (with Start at a smaller size, that size too, since the boost runs early in the run); below that the tick is greyed out. The model sees a picture as 16-pixel patches, and in a medium or full-length shot a face at 0.5 MP can be just one to three patches wide: boosting it teaches the LoRA pixelation. It's for sets where the boosted part (the face, the clothes and so on) is small in the frame.

**What it boosts:** Face and hair (the default) or Face only come from the face finder; Face only suits a subject whose hair changes between pictures. The other choices are the same as Make masks' person finder (the clothes, the body, the background and so on), found with the same model. The clothes choice suits an outfit LoRA, not a person one. [Masked training](MASKED_TRAINING.md) explains masks.

## Sub-folders

A training folder is read at its top level only. Tick **Also train on its sub-folders** on the Training tab (off by default) and the pictures in its sub-folders train too, so a big dataset can stay organised in folders: outfits, forms, sources, or the thousands of pictures of a fine-tune. The folder you choose on the Start tab stays the training folder; the tick only adds what is inside it.

**What it trains.** Every sub-folder holding pictures (or clips or songs) is found, at any depth. Each trains as its own set beside the training folder, with its own captions next to its pictures and its own cache, so two folders can hold pictures with the same name. Every picture trains once an epoch, wherever it is: a folder with more pictures carries more weight, and nothing is evened out between folders, unless you use repeats (below). The training folder itself can hold no pictures at all, with everything in its sub-folders.

**What it leaves out.** Fizgig's own folders, matched in any case (Image Prep's `originals`, and `originals_2`, `originals_3`… from later runs; the Look Consistency Filter's `excluded_by_look`; Find duplicate pictures' `excluded_duplicates`; and `removed`, where Remove puts pictures), hidden folders (a name starting with `.`), the LoRA output folder and the cache folder when they sit inside, and a fine-tune's regularisation folder (below), with everything under them. The run's log names the Fizgig folders it left out.

**The line under the tick** says what it will do before you press Start:
- Unticked, when there are sub-folders: how many there are and how many more pictures they hold, which aren't trained on.
- Ticked: how many folders and pictures will train, and first the folders whose pictures aren't captioned yet (the training folder's own too), with how many pictures each. **Show all folders** lists every folder with its count.
- Start refuses a sub-folder whose pictures have no captions and names it, as it does for the training folder.

**Repeats from folder names.** With the sub-folder tick on, a second tick, **Repeats from folder names** (off by default), reads a number at the start of a sub-folder's name as how many times it trains an epoch: `10_Laffey_base` trains 10 times, `2_Laffey_casual` twice, and a folder without a number once (the training folder itself always once). It's the naming kohya and OneTrainer use, so a dataset laid out for them works as it is. Use it to give a small folder as much of each epoch as a big one, or to make one folder count more: 50 pictures at `2_` weigh the same as 100 at `1_`. The number is read from the folder's own name, from 1 to 100 (`0_` trains once, above 100 trains 100 times). Unticked, a number in a name means nothing; when your folder names look like repeat counts, the line under the sub-folder tick says so.

The line under the repeats tick is a table: each folder's pictures, its repeats, how many it is seen per epoch and its share of training, largest first. Under it: what the repeats do to your run (an epoch's length, and the Max Epochs that keeps the run about as long as before). When no folder is numbered yet it suggests a rename that evens out the smallest folder; when the names already look like repeat counts, the line under the sub-folder tick offers a button that ticks both. A folder with under 3% of an epoch is marked, so a `1_` typed for `10_` shows before you start. Repeats make an epoch longer (and so the previews every N epochs further apart); a regularisation set on **Match my training pictures** follows the longer epoch. The run's log has a `[repeats]` line, and the LoRA's metadata records each folder's repeats (`ss_folder_repeats`).

**Captions and Image Prep** work on one folder at a time, and their lines say so when the training folder has sub-folders. To caption a sub-folder, make its masks, resize it or run the Look Consistency Filter on it, choose it as the training folder on the Start tab, then choose the top folder again to train. Each sub-folder's trigger word is the one you saved for it on the Captions tab, else the word its captions start with. With masked training on, each sub-folder's pictures use the masks in their own folder.

**While it trains** (the Krea 2 loss watch, [details](KREA2.md#the-trainer-curates-your-dataset-while-it-trains-experimental)), every picture is kept with its own folder:
- **Per-image loss** (on the run card while it trains, and on the run in Runs & outputs) names each picture with its folder (`outfits/red/03`), and a caption you edit there is written into that picture's own folder.
- **The loss watch on the Captions tab.** Each caption card shows its picture's loss-watch verdict (stuck, suspect, learning, easy and so on, coloured as in the per-image loss view) from the run training on that folder, or the folder's last run. A caption saved on the Captions tab while a run trains on that folder trains from the next epoch: the trainer re-encodes it at the epoch boundary, as an edit from the per-image loss view, with or without the loss-watch ticks on. In the per-image loss view, **Edit on the Captions tab** opens the tab on that picture.
- **Auto-recaption** writes the new caption into the picture's own folder, with that folder's trigger word.
- **Exclusions** are saved in each folder's own `fizgig_excluded.json`, so they travel with its pictures.
- The **look-outlier warm-up** reads each folder's own Look Filter scores (`fizgig_look_scores.json`, from a scan of that folder). Pictures moved into a sub-folder's `excluded_by_look` are left out.
- Pause and Resume carry all of it on, and **Load settings from last train** brings the tick back.

**Fine-tunes** train every folder in the tree, which is the main use: a dataset of thousands of pictures sorted into folders. If the folder you picked for **Regularisation images** sits inside your training folder, its photos are not trained as your own pictures: they are only used as regularisation images, as they would be from any other folder.

**On a pod or another computer** the tick decides what goes:
- **Upload a folder** on the Start tab keeps the sub-folders as they are when it is ticked. Unticked, the files of the sub-folders all go into the one new folder.
- **Send to this pod** (the Fizgig window's **From this computer** card) sends the sub-folders too when it is ticked; each tile says whether its sub-folders go ("sent too" or "not sent"). Sending again sends only what isn't there yet.
- **Download a training folder to this pod** (Start tab, on a pod) unpacks a dataset with its sub-folders as they are. Tick the box to train them all, or press **Choose…** after **Train on this folder** to pick one sub-folder (it opens at the downloaded folder).
- **Move a run** takes the sub-folders with the run, and captions edited over there come back into each folder. See [Move a run](MOVE_RUN.md).

**Not with** Subjects 2-4, DOP or Fizgig LoRA Isolation, which keep one folder per subject, nor Edit or slider training, which pair their files: the tick is hidden in those modes. To train several people, use Subjects; to train one subject's outfits or forms, use sub-folders or one folder with the outfit named in each caption.

For the command line, see [CLI](CLI.md): each sub-folder is a `[[datasets]]` block with an `item_prefix`.

## Several captions per picture

A caption file normally is one caption: if it has several lines, they are joined and train as one long caption. Tick **Several captions per picture (one per line)** in Training Parameters and each line is a caption of its own: every time a picture comes round, one of its lines is picked at random. Use it for a line of booru tags plus a sentence, so the LoRA answers to either kind of prompt, or for captions written by several captioners, so no one bad caption dominates. OneTrainer writes multi-caption files the same way, so its datasets work as they are.

The tick only shows when your caption files have more than one line (or it's already ticked). Its note says how many of your files do; ticked, it counts the pictures and captions, and the cost: caching encodes every line, so it takes longer in proportion, while training runs at the same speed. Blank lines are ignored. Start's trigger-word check looks at each line. Features that need a single caption use the first line: **Leave the rest of the model alone**, training maps, and the multi-person sample caption. A caption you fix in the Problem Images window re-encodes every line; **auto-recaption** replaces the first line only and keeps your other lines. It is for still-picture LoRAs and fine-tunes on every family except MiniMax H3 and ACE-Step, and not for edit or slider training. The run's log has a `[captions]` line, and the LoRA's metadata `ss_multi_caption`.

Put the caption style you prompt with most on line 1. Unticked, each file trains as one caption, whatever its lines. Ticking or unticking it between Pause and Resume caches the captions again before the run carries on.

On the Captions tab, each line of a caption shows in its own box, every other one shaded and numbered once there are two (Enter starts a new line, Backspace at the start of a line joins it to the one above); a long caption scrolls in its box. To build two-line files, caption with one model, then tick **Add as another line** and caption with another: each file keeps its caption and gets the new one on a line of its own (a line already there isn't added twice; captioning again with the same model adds a second, slightly different line, and the notes under the tick say so). While it's ticked, Overwrite has no say, and a static or song caption writes only the files without one. With the tick on and Overwrite on, a note says how many files hold several captions, which Overwrite would replace with one. With the tick on, a card's **Re-caption** on a file of several captions rewrites its first line and keeps the others; with it off, Re-caption replaces the whole file, as it always has. **Find, Replace & Add** offers **Add to every line** when the folder has multi-line files; unticked, the words go at the start or end of the caption (on a caption of several lines: the start of the first line, the end of the last). **Translate captions** translates each line on its own, keeping its trigger word; with Skip captions that already contain Chinese ticked, a line added since the last translation is translated and the lines already done are left as they are.

## Subjects: more than one person, and keeping them to their trigger

The Training tab's **Subjects** card says who the LoRA is of and how it behaves with other people. 1 subject with Off is ordinary LoRA training. Every other setting is Fizgig's own techniques combined with DOP (Differential Output Preservation, from @ostris's AI-Toolkit).

- **Who is this LoRA of?** 1 to 4 subjects, each a folder of their photos with their own trigger word in every caption. Subject 1 is the Start tab's folder.
- **Your subject and other people:**
  - **Off - a normal LoRA:** your subject's face can leak onto other people, even in prompts that don't use its trigger word.
  - **Leave the rest of the model alone:** ask for "a man at a party" without the trigger word and you get a stranger, just as the model made before you trained. With one person, use it when your subject must stay out of prompts without the trigger word; otherwise use Off. **Fizgig LoRA Isolation** needs two or more subjects: with one, it's greyed out. **Subject 1 - what they are** (one word: man, woman, dog) takes the trigger word's place in the captions that keep the model as it was, so with captions that don't say what the subject is (`slcp walking`) the model's own `a man walking` is what stays. Left empty, the trigger word is just left out (`walking`); a caption that already says it (`slcp, a man walking`) keeps `a man walking` either way.
  - **Fizgig LoRA Isolation:** your subjects can stand together in one picture, or next to someone the model already knows, and every face stays right. Best for people; about 1.5x the training time; its 21 MB of strangers come with any model's download in Preferences. It joins each trigger to what the subject is (`zwxem, a woman` trains as `zwxemwoman`, your caption files untouched) and adds 16 side-by-side pictures per subject next to fixed strangers. **Use my captions exactly as written** turns the joining off for anyone who writes the joined word themselves.
- At the bottom of the card, each subject shows one of its own captions and exactly what training makes of it, and it updates as you change the settings above.
- Two subjects of the same kind (two women) need about 1.6x the epochs; the Training tab says so.
- Up to 4 people work well, 2 or 3 of them in one picture. With a person the model already knows well in the picture too, 2 subjects is reliable and 3 is hit and miss: see [How many subjects](MULTI_PERSON.md#how-many-subjects).

How to caption for it, choose trigger words and prompt the finished LoRA: [People in group pictures](MULTI_PERSON.md).

### Regularisation dataset

For a LoRA that changes too much besides its subject: plain prompts drift toward your subject, or the whole style shifts. The fold at the bottom of the Subjects card, under **Check a trigger word**, trains a second folder alongside your training pictures: ordinary pictures of the same kind of thing (other people for a person, other characters for a character, other pictures in the genre for a style). They hold the model's idea of that kind of thing in place while your subject trains. It works for any subject, anime and drawn characters included, where Fizgig LoRA Isolation suits people best.

- **Train with a regularisation dataset** turns it on. The rest of the Subjects card is set aside for that run: one subject (the training folder), no DOP, no Fizgig LoRA Isolation. Those choices are greyed out and keep their values, so unticking brings them back.
- **Regularisation pictures:** real pictures, not ones the model made (those teach it its own mistakes). Every picture needs a caption that describes it and does not use your trigger word; Start refuses pictures without one and warns about captions that use the trigger. Caption them by making that folder the training folder on Start for a moment, then switching back.
- **Regularisation LR ×:** the share of the learning rate the regularisation pictures train at. 1.0 (the default) trains them as firmly as your own pictures; lower makes them a lighter anchor.
- **Use each epoch:** how many of the folder's pictures each epoch trains on, a different handful each epoch, so over a run the whole folder is used. **Match my training pictures** (the default) takes as many as your training pictures; or a share of the folder from 10% to 100%. The line beside it says how many that is and how long an epoch becomes.
- Standard LoRA and LoKR, on every still-picture model. A fine-tune has its own regularisation folder on its card. While it is on, a run can't be shared with a helper GPU (Move a run still takes the folder with it).
- Regularisation pictures are never judged, throttled, excluded or recaptioned by the per-image loss watch, never masked or face-boosted, and never used for training maps.

## Output folders

The LoRA output folder lives on the Training tab's Output section and is remembered per model: switch models and each gets its own folder back. The default is `output_loras` inside Fizgig.

## Compatibility

Loads kohya, PEFT, OneTrainer (OMI + legacy), AI-Toolkit, and LyCORIS (LoKR / LoHa) — auto-converted, and LoKR/LoHa run natively everywhere: Repair Studio, Profiler, Extract, Context LoRA. Repair Studio saves LoKR as LoKR, losslessly. Output is `.safetensors` that drops straight into ComfyUI.

## The workbench

Each tool works on a trained run's output **or any LoRA you've downloaded** — and they hand off to each other (profile → repair → explore → compare, one closed loop). MiniMax H3 previews render a short clip, judged by its middle frame — the model's native regime.

### Repair Studio

A live slider per transformer block (32 on Klein, up to 50 + the token refiners on MiniMax H3) with a side-by-side preview that updates as you drag. **Reuse unchanged blocks** caches per-block activations so late-block edits redraw up to 97% faster; the baked save is always exact. Blend blocks from a second **donor** LoRA, balance the pair per block, condition previews on a reference photo, and save a `.safetensors` that works in ComfyUI at strength 1.0. On MiniMax H3 the previews are **clips with sound**, played side by side in the app, with a per-block library and first/last-frame conditioning — see [Repair Studio on H3](MINIMAX_H3.md#repair-studio-on-h3). Krea 2 has its own **✨Text fusion** presets — see [KREA2.md](KREA2.md#help-map-krea-2s-blocks).

### Exploring variations

Repair Studio's **🎲 I'm feeling lucky** rolls a new strength for each of the LoRA's blocks between your Min and Max (locked blocks keep theirs; Hold keeps each part of the model in its own safe range), renders it, and leaves the sliders where it landed: keep a roll you like, roll again or Undo, lock the blocks you like and roll the rest, or nudge single blocks from there. It does what the old desktop app's LoRA the Explorer did, on the same page as the fixing and saving. A **load strength** box sets the strength the LoRA is meant to be used at.

### LoRA Royale

Point it at a training run and it renders **every epoch on one fixed seed**, with a crossfade slider — drag until it looks best and stop. An optional **likeness score** (the Image Prep tab's look match, CPU) rates each epoch against a training photo and jumps you to the best. Then make it shareable: epoch-morph clips, seed / prompt / strength **travels**, a **comparison sheet** (with/without-LoRA grid, same seed per row), all exportable as looping MP4/GIF with an optional deflicker pass. Works on any folder of LoRAs, or a single file.

### Profiler

How much rank a LoRA really uses and where its weights are, then renders with each block group switched off and scores likeness and bleed — which blocks carry the subject and which leak it. The result is a colour-coded HTML report. Repair Studio reads its sidecar automatically and shows the findings inline when you load the same LoRA. Klein's block map is in [KLEIN.md](KLEIN.md#block-map).

### Extract

Distil any Klein, Flux 3 Image Klein, Fibo 2.0, Krea 2, MiniMax H3 or Qwen Image 2.1 LoRA to a lower rank — it runs weight-only SVD with no models loaded, and Klein's presets keep just the Identity, Style+Composition or Details blocks (or any blocks you pick). PEFT and LyCORIS sources supported.
