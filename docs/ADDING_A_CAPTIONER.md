# Adding a captioner

A captioner is a model that writes a picture's caption on the Captions tab. Like a diffusion model's driver, it comes into Fizgig by speaking Fizgig's language: a description of what it is, plus a small class with three jobs. Fizgig does everything else.

## The rule that keeps training safe

**A captioner runs only in the Captions tab's own worker process** (`fizgig.scripts.batch_caption`), which exits to free its memory. It never loads in a training run, so no training memory plan, block swap or Studio plan has to know it exists.

Auto-recaption (inside a run) stays on the one captioner the training plans already fit around: Krea 2's Qwen3-VL-4B, or a secondary captioning file of the same architecture. It calls that model directly, not through the registry. `checks/test_captioners.py` fails if any training code imports `fizgig.captioners`. A pull request that changes this rule won't be merged.

## What you write

Two things, and nothing outside `src/fizgig/captioners/`:

**1. A description** in `src/fizgig/captioners/registry.py`:

```python
CaptionerDescription(
    key="wd_eva02_large_v3",                 # stable id, lowercase
    label="WD EVA02 Large v3 (booru tags)",  # what the model menu shows
    kind="tags",                             # "tags" or "sentences"
    module="fizgig.captioners.wd_tagger:WDTagger",
    files=HFRepo("SmilingWolf/wd-eva02-large-tagger-v3", "<40-character commit>"),
    licence="Apache-2.0",
    vram_gb=1.5,                             # what it needs on the card
    size_gb=1.3,                             # its download
)
```

- **`files`** is pinned to a commit (`revision`, plus `code_revision` for `trust_remote_code` models), so a change upstream can't swap what runs. Use `PrefFile` only for a file Preferences already holds.
- **`tasks`** / **`default_task`**: a fixed menu the model understands (Florence's `<CAPTION>` …).
- **`instruction=True`**: it follows a written instruction. It then gets the Captions tab's instruction menu and edit box. A tagger never does.
- **`vram_gb`** is what it needs at its smallest, checked before a caption job starts beside a running run. Measure it, don't guess.
- **`prompt_types`** / **`prompt_lengths`** / **`prompt_extras`**: an instruction captioner's own prompt menu (JoyCaption's caption types, lengths and extra options). Types are `(label, template, template with {word_count}, template with {length})`; extras are `(label, sentence, ticked by default)`, with `{name}` for the trigger word. The Captions tab shows them under Fizgig's own tasks and builds the instruction; your class just receives it.
- **`precisions`**: for a model that loads more than one way, `(("bf16", 19.0), ("nf4", 8.0))`, best first. The worker picks the first that fits the card's free memory and hands it to your class as `self.precision`.

**2. The class**, in its own file in `src/fizgig/captioners/`:

```python
from fizgig.captioners import Captioner, CaptionRequest, CaptionResult

class WDTagger(Captioner):
    def load(self):
        ...   # self.files: {"repo", "revision", "code_revision"} (or {"path"}); self.device: "cuda" / "cpu"

    def caption(self, image, request: CaptionRequest) -> CaptionResult:
        ...   # image: one upright RGB PIL picture (a clip's middle frame; see-through laid on white)
        return CaptionResult(tags=[("1girl", 0.97), ("long_hair", 0.91)])   # or CaptionResult(text="...")

    def unload(self):
        ...
```

`request` carries `task`, `instruction` and `max_new_tokens`.

Preferences lists every captioner that downloads itself, with its `blurb` and `size_gb` and a **Download** button. That button runs your class's `fetch(desc, files)`, which by default downloads the pinned snapshot minus `download_skip` (list there any second copy of the weights your class never loads). If your model's first use caches more than its own files, override `fetch()` to do that too, so it works offline afterwards. For example, a `trust_remote_code` model's code, which can live in another repo: see `FlorenceCaptioner.fetch`. Import heavy libraries inside `load()`, not at the top of the file: the registry is read by the web server, which must not load torch or model code to show a menu.

## What Fizgig does, so your class doesn't

- Downloads, the worker process, batching, progress, Stop.
- Overwrite, detail crops, which files to caption.
- Opening the file: orientation, clips, transparency.
- The trigger word, always first.
- For tags: the score threshold, tags to leave out, underscores to spaces, best first.
- Writing the caption file.

A captioner never writes a file, reads Preferences or sees a folder.

## Checking it

```bash
python -m pytest checks/test_captioners.py
python -m fizgig.scripts.batch_caption --captioner wd_eva02_large_v3 --list_file images.txt --trigger zwxem
```

The checks confirm your description is complete and pinned, and that your class has the three methods. The command captions every file listed in `images.txt` (one path a line). Include in your pull request a few of its captions, and the VRAM you measured.
