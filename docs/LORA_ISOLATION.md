# Fizgig LoRA Isolation

[← Back to the README](../README.md) · [How to caption for it](MULTI_PERSON.md) · [Demo video](https://youtu.be/CNh4sN5vN_8) · [How to set it up](https://youtu.be/8uW5NdL5b20)

Fizgig LoRA Isolation trains two or more people into a LoRA so that everyone else in a picture keeps their own face. A normal character LoRA spreads its subject's likeness onto other people: strangers in the background, a friend standing next to them, a crowd. With LoRA Isolation the subjects can stand together in one picture, or next to a person the model already knows well, and every face stays its own. Without the trigger word you get the model's own people, as before training.

It builds on DOP (Differential Output Preservation) from Ostris's AI-Toolkit and adds the layers described below. In Fizgig it's set up on the Training tab's **Subjects** card and works on every model except ACE-Step. It has been tested on Krea 2; more model testing is to follow.

## Results

Two LoRAs were trained on Krea 2: same photos, same settings, same seed, same epoch, and one difference, Fizgig LoRA Isolation. Both learned two people: the subject and a friend as a second character (plus a paper-cut style). LoRA Isolation is built for two or more people, and it protects every one of them. Each LoRA was rendered on 25 prompts at two seeds (50 pictures). Every prompt is a split side-by-side picture: on one side a different, carefully described white man (age, hair, facial hair, eyes, skin, glasses, clothes, expression), on the other side the trained subject, also a white man. This is the hardest case for leakage. Each half was scored with ArcFace (insightface `buffalo_l`) against the subject's mean face from 35 training photos. The same person typically scores 0.45 or more; unrelated people score 0.0 to 0.2.

| | Normal LoRA | Fizgig LoRA Isolation |
|---|---|---|
| Strangers that read as the subject | 48 of 50 | 0 of 50 |
| How much the subject looks like themselves | 0.77 | 0.78 |

Likeness doesn't drop. For scale, the subject's own real photos score 0.81 against each other; these renders reach 0.78, in a LoRA shared with a second person. A second person in a LoRA is usually the problem. Here it's part of the fix.

Leave the trigger out ("a man") and the subject is gone: you get the base model's own people.

Even a famous face the model knows by heart, put right between two trained subjects, stays exactly themselves, with no trace of either subject.

## How it works

Six parts, all applied by Fizgig when **Fizgig LoRA Isolation** is chosen on the Subjects card.

### 1. DOP: Differential Output Preservation (Ostris, AI-Toolkit)

Each training picture also trains with its trigger removed from the caption, and on that pass the LoRA is held to the base model's own prediction for the same noised input. Without the trigger, the LoRA learns to change nothing, so the subject stays out of prompts that don't name them.

### 2. Multi-subject identity anchoring

Two or more identities train together, each from its own folder with its own trigger. Two faces learned side by side teach the LoRA that a face belongs to one trigger, rather than to the class in general. This is why a second person helps instead of hindering, and it works across genders: the test above used a woman as the second subject, and every stranger was a man. Even for a LoRA of one person, training a second person alongside them is recommended.

### 3. Fused-token class decoupling

The trigger and its class word are merged into one word in every caption: `gwelt, a man` trains as `gweltman`. The plain class token (" man") is the one every other man in the model shares, and it's the bridge a subject's features cross to reach other people. Joined onto the trigger, the class word becomes a different token, so the subject's look never attaches to the shared one. DOP's trigger-free pass then replaces the fused word with the class phrase (`gweltman` → `a man`).

Fizgig applies the join while the captions are read, through the dataset's `caption_replace`. The caption files on disk are never changed. A trigger must join cleanly with its class word, or the class token is no longer whole: `fyxo` + woman reads as `f | yx | ow | oman`. The Subjects card's **Check a trigger word** tests this with the training model's own tokenizer.

### 4. In-frame negative-pair compositing

Each person gets 16 side-by-side pictures: one of their own training photos next to a stranger, in a single 2:1 frame, with the subject's side chosen at random. These teach "this person is the subject, that person stays a stranger" inside one picture, which is exactly where a normal LoRA leaks.

The subject's half is a face, so the photos are picked for one. Image Prep's face crops (the `FaceCrop_` files) go first and are used as they are, never cropped further. If there are fewer than 16, photos where a face is found come next, each cut to the largest square with the face as central as the photo's edges allow, never zoomed. Photos with no face (an outfit detail, a back view) are used only when nothing else is left. No photo is used twice until all the usable ones have been. To pick from every photo instead, for more of the body shape, untick **Side-by-side pictures: face crops first** on the Subjects card; a face crop that's picked is still never cropped.

The captions describe the stranger first and the subject second: `an Indian man in his 30s on the left, and gweltman on the right, a split-screen diptych of two separate close-up portraits`. The text encoders these models use are causal, so every word after the trigger is encoded with the trigger already seen; putting the stranger's description first keeps it clean of the subject. The ending says what the picture is, two separate portraits joined, in words a prompt with your subjects together doesn't use. With the old ending, "two people side by side", two subjects borrowed each other's eyes and clothes more often.

The stranger's half of each side-by-side is masked: the stranger counts, their background doesn't, even when **Train on the parts your masks show** is off. The masks come with the strangers set. Your subject's half trains whole, as before, so you don't need to make masks for it.

The strangers are a fixed set of 32 generated people, each with a person mask: 8 ethnicities, men and women, aged 20s to 50s. Version 2 is rendered on Qwen Image 2.1 for natural skin texture, so the set doesn't pull a LoRA towards poor skin, and no real person's photo is used. It's published as [ShootTheSound/fizgig-multi-person-strangers](https://huggingface.co/datasets/ShootTheSound/fizgig-multi-person-strangers) (CC BY 4.0), pinned to a revision, and downloads with Fizgig's models.

### 5. Per-sample LR scaling, strided DOP and balanced sampling

- The side-by-side pictures train at 2× the learning rate on their own steps (`--lr_boost fizgig_mp_=2`).
- DOP runs every second step at 2× weight: the same average preservation for about half the extra compute.
- **Balance the subjects** repeats smaller folders so each subject trains about as often as the largest.

### 6. Class-aware training paths

Each subject is a person, an animal or thing, or a style or look, read from its captions or set on the card:
- **A person** gets the fused trigger and the side-by-sides.
- **An animal or thing** gets the fused trigger (`rexodog`) and no side-by-sides.
- **A style or look** keeps its trigger as written and gets no side-by-sides, because a style should change the whole picture. DOP drops the trigger from the trigger-free caption, so the look only appears when it's called.

## The recipe used for the results above

| Setting | Value |
|---|---|
| Model | Krea 2, Standard LoRA |
| Subjects | the subject (36 photos, a person), a second person (53 photos, a person), a paper-cut style (8 pictures, a style), balanced |
| Rank / alpha | 16 / 16 |
| Learning rate | Adaptive LR, 2e-4 to 4e-4, constant scheduler |
| Optimizer | AdamW 8-bit, max grad norm 1, batch 1 |
| EMA | 0.98 |
| Resolution | 0.25 MP (496 × 496), bucketed, no upscale |
| Base precision | INT8 |
| Seed | 42 |
| Checkpoint used | epoch 20 |
| LoRA Isolation | 16 side-by-sides per person at 2× LR, DOP every second step at weight 2 |

The normal LoRA it was compared with learned the same two people and style, from the same pictures in one folder, with the same per-epoch mix of copies, the same triggers as written, the same settings, seed and epoch, and none of LoRA Isolation's parts.

Training time is about 1.5× a normal LoRA.

## What's known so far

- **How many people:** up to 4 subjects train well. Two trained people in one picture, plus a person the model already knows, works almost every time. Three or more trained people in one picture is hit and miss.
- **Same-class subjects** (two women) need about 1.6× the epochs. Both feed the same "a woman" preservation pass, so that class is held roughly twice as hard.
- **Prompting:** use the joined word (`gweltman man at a cafe`), and describe the other people in the picture (hair, age). The LoRA keeps them apart on its own; the description is a safety net.
- **Tested** on Krea 2. The method runs on every Fizgig model except ACE-Step; on MiniMax H3 the subjects train from their photos (clips and voice files in their folders are left out).

## Using it

On the Training tab, the **Subjects** card: choose how many subjects, pick each one's folder and what it is, and set **Your subject and other people** to **Fizgig LoRA Isolation**. Under each subject the card shows exactly what training will use. Captioning guide: [MULTI_PERSON.md](MULTI_PERSON.md). Command-line flags: [CLI.md](CLI.md#keeping-a-subject-to-its-trigger-dop).

## Credits

Thanks to Ostris for DOP (Differential Output Preservation, from [AI-Toolkit](https://github.com/ostris/ai-toolkit)), one of the core components this method builds upon. Likeness scoring uses [insightface](https://github.com/deepinsight/insightface) (`buffalo_l`). The strangers set is rendered with Qwen Image 2.1.
