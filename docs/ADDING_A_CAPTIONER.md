# Adding a captioner

> **Scope of a captioner pull request.** It adds one description to `src/fizgig/captioners/registry.py` and one file of its own in `src/fizgig/captioners/`. Nothing else. If your captioner needs something the contract doesn't have, add it as an optional description field or `Captioner` method whose default leaves every other captioner exactly as it was, and say why in the pull request. A fix or improvement to shared code (the worker, the Captions tab, Preferences) goes in its own pull request. Fizgig is maintained by one person; that's only possible if adding one captioner can't change the others.

A captioner is a model that writes a picture's caption on the Captions tab. Like a diffusion model's driver, it comes into Fizgig by speaking Fizgig's language: a description of what it is, and a small class with three jobs. Fizgig builds everything else from those two pieces:
- the model menu, task menu and option rows on the Captions tab;
- its row, with a Download button, in Preferences' Captioning models card;
- the worker process, batching, progress and Stop;
- the memory check beside a running run.

## Before you start: open a Discussion

Adding a captioner starts with a post in the repo's [New models & fine-tuning](https://github.com/shootthesound/Fizgig/discussions/new?category=new-models-fine-tuning) discussions, before any code: which model, where its weights live, its licence, what it's better at than the captioners Fizgig already has, and how much memory it needs. Discussions are for that only; bugs, questions and feature requests go in [Issues](https://github.com/shootthesound/Fizgig/issues). It's a prerequisite for a captioner pull request: it lets us agree it's worth adding, and avoids two people building the same one.

## The rule that keeps training safe

**A captioner runs only in the Captions tab's own worker process** (`fizgig.scripts.batch_caption`), which exits to free its memory. It never loads in a training run, so no training memory plan, block swap or Studio plan has to know it exists.

Auto-recaption (inside a run) stays on the one captioner the training plans already fit around: Krea 2's Qwen3-VL-4B, or a secondary captioning file of the same architecture (Qwen3-VL 8B). It calls that model directly, not through the registry. `checks/test_captioners.py` fails if any training code imports `fizgig.captioners`. A pull request that changes this rule won't be merged.

## The existing captioners

Read them side by side; between them they use every part of the contract.

| Captioner | File | Worth reading for |
|---|---|---|
| Florence-2 (three) | `captioners/florence.py` | A fixed task menu (`tasks`); `trust_remote_code` with pinned code, including code that lives in another repo, and the `fetch()` override that caches it |
| Qwen3-VL | `captioners/qwen3vl.py` | A file Preferences holds (`PrefFile`) rather than a download; following an instruction |
| JoyCaption Beta One | `captioners/joycaption.py` | Two `precisions` (bf16, 4-bit NF4) picked by free memory; its own prompt menu (`prompt_types`, `prompt_lengths`, `prompt_extras`) |
| WD EVA02 tagger | `captioners/wd_tagger.py` | `kind="tags"`: tags with scores, which Fizgig thresholds; `download_skip` for weights the class never loads |

## What you write

**1. A description** in `src/fizgig/captioners/registry.py`:

```python
CaptionerDescription(
    key="wd_eva02_large_v3",                 # stable id, lowercase
    label="WD EVA02 Large v3 (booru tags)",  # what the model menu shows
    kind="tags",                             # "tags" or "sentences"
    module="fizgig.captioners.wd_tagger:WDTagger",
    files=HFRepo("SmilingWolf/wd-eva02-large-tagger-v3", "<40-character commit>"),
    licence="Apache-2.0",
    vram_gb=2.0,                             # what it needs on the card (at its smallest)
    size_gb=1.3,                             # its download
    blurb="Booru tags, for anime and illustration",   # Preferences' row
    download_skip=("model.onnx", "model.msgpack"),    # copies of the weights the class never loads
)
```

- **`files`** is pinned to a commit (`revision`, plus `code_revision` for a `trust_remote_code` model), so a change upstream can't swap what runs. Use `PrefFile` only for a file Preferences already holds.
- **`tasks`** / **`default_task`**: a fixed menu the model understands (Florence's `<CAPTION>` …).
- **`instruction=True`**: it follows a written instruction. It then gets Fizgig's instruction tasks and edit box. A tagger never does.
- **`vram_gb`** is what it needs at its smallest, checked before a caption job starts beside a running run.
- **`precisions`**: for a model that loads more than one way, `(("bf16", 19.0), ("nf4", 8.0))`, best first. The worker picks the first that fits the card's free memory and hands it to your class as `self.precision`.
- **`prompt_types`** / **`prompt_lengths`** / **`prompt_extras`**: an instruction captioner's own prompt menu (JoyCaption's caption types, lengths and extra options). Types are `(label, template, template with {word_count}, template with {length})`; extras are `(label, sentence, ticked by default)`, with `{name}` for the trigger word. The Captions tab shows them under Fizgig's own tasks and builds the instruction; your class just receives it.
- **`blurb`** and **`size_gb`**: Preferences lists every captioner that downloads itself with these and a Download button.

**2. The class**, in its own file in `src/fizgig/captioners/`:

```python
from fizgig.captioners import Captioner, CaptionRequest, CaptionResult

class WDTagger(Captioner):
    def load(self):
        ...   # self.files: {"repo", "revision", "code_revision"} (or {"path"}); self.device: "cuda" / "cpu";
              # self.precision: one of the description's precisions, or ""

    def caption(self, image, request: CaptionRequest) -> CaptionResult:
        ...   # image: one upright RGB PIL picture (a clip's middle frame; see-through laid on white)
        return CaptionResult(tags=[("1girl", 0.97), ("long_hair", 0.91)])   # or CaptionResult(text="...")

    def unload(self):
        ...
```

- `request` carries `task`, `instruction` and `max_new_tokens`.
- Import heavy libraries inside `load()`, not at the top of the file: the web server reads the registry to build its menus and must not load torch or model code to do it.
- **`fetch(desc, files)`** is what Preferences' Download button runs. By default it downloads the pinned snapshot minus `download_skip`. If your model's first use caches more than its own files, override it to do that too, so the captioner works offline afterwards. For example, a `trust_remote_code` model's code, which can live in another repo: see `FlorenceCaptioner.fetch`.

## What Fizgig does, so your class doesn't

- Downloads, the worker process, batching, progress, Stop.
- Overwrite, detail crops, which files to caption.
- Opening the file: orientation, clips, transparency.
- The trigger word, always first.
- For tags: the score threshold, tags to leave out, underscores to spaces, best first.
- Writing the caption file.

A captioner never writes a file, reads Preferences or sees a folder.

## Before you ship

Each of these is something a reviewer will check. Put the results in your pull request.

**The description**
- [ ] `files` pinned to a 40-character commit (and `code_revision` if it runs its own code).
- [ ] The licence named, and one that allows people to download and use the model as Fizgig does.
- [ ] Not gated, or the pull request says why it's worth asking people for a Hugging Face token.
- [ ] `blurb` in a few plain words, `size_gb` as the real download (after `download_skip`).

**Measured, on a real card**
- [ ] Peak VRAM at each precision (`torch.cuda.max_memory_allocated()` over a load and a few captions), and `vram_gb` / `precisions` set from it, not guessed.
- [ ] If it's too big for a 16 GB card in full precision, a smaller precision that fits one, measured.
- [ ] Seconds per picture once loaded.

**Downloads and offline**
- [ ] Download from Preferences into an empty Hugging Face cache, then caption with the network off: `HF_HUB_CACHE=<empty folder> python -m fizgig.scripts.batch_caption --fetch <key>`, then the same `HF_HUB_CACHE` with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and `--captioner <key> --list_file images.txt`. This catches files the first use caches that a plain download misses.
- [ ] The same from a first use (no Download), then offline.

**Captions**
- [ ] A varied set of at least ten pictures (people, an object, a style or illustration, a transparent PNG) captioned by yours and by Qwen3-VL with the same trigger word, both sets in the pull request.
- [ ] For an instruction captioner: Fizgig's Training caption task, and one of your own `prompt_types` if it has them.
- [ ] For a tagger: the threshold you'd suggest, and how it handles rating and character tags.

**Checks and docs**
- [ ] `python -m pytest checks/test_captioners.py` passes, and so does the full `checks/` suite.
- [ ] `docs/WEB_UI.md` (the Captions tab's list), `docs/INSTALL.md` (first-use download sizes) and `docs/CLI.md` (its `--captioner` key) updated, in plain words for a user.

## The pull request

- A link to the Discussion where it was agreed.
- The two files, and nothing else (see the scope note at the top).
- The results from Before you ship: VRAM per precision, seconds per picture, the offline test, and the caption comparison.
- Who made the model, credited by their handle.
