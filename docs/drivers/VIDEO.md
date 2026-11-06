# Adding a video model

A video family trains on photos and clips (and, if the model has an audio stream, sound). Everything in [STILLS.md](STILLS.md) applies; this page covers what clips add. MiniMax H3 (`families/minimax.py`, `minimax/driver.py`) is the complete example: photos, clips and voice recordings, with sound.

The layer is built so a video model needs a description and a driver, the same as a stills model: the dataset finds and checks clips from the description, the trainer hands clips to your `training_loss`, and the workbench has a generic video engine that renders clips through your driver.

## 1. The description

Three fields turn clips on:

```python
from fizgig.families.description import ClipSpec

MYVIDEO = FamilyDescription(
    ...,
    media=("photo", "clip"),          # add "voice" if the model also trains on sound alone
    clip_spec=ClipSpec(
        fps=24,                       # the model's clock: a clip at another rate is refused, never resampled
        frame_step=8, frame_offset=1, # the frame counts the VAE encodes: frame_offset + n * frame_step ...
        max_frames=121,               # ... up to this
        edge_multiple=32,             # width and height
        audio_rate=0,                 # sound: its sample rate (0 = the model has no audio stream)
        audio_channels=2,
        mute_suffix="_mute",          # a clip named ..._mute trains its pictures only
    ),
    clip_regimes=(("dial", 4, 1.0), ("confirm", 8, 0.75)),   # optional, see the workbench below
)
```

What each one does:

- **`media`**: the dataset globs `.mp4` files (and sound files with `"voice"`) only for a family whose media include them. The Training tab's clip controls (Clip Target Megapixels, any of your options with `show_if_media="clip"`) appear when the training folder holds clips.
- **`clip_spec`**: every clip is checked against it twice, once by the launch before anything caches (Start lists each off-spec clip with the value found and what to do) and again when the dataset decodes it. Fizgig never transcodes, resamples or trims: a 30 fps clip trained as 24 learns motion at the wrong speed and looks fine doing it, so it is refused instead. Gizmo, Fizgig's clip tool, prepares clips to a spec.
- **`clip_regimes`**: named render settings for the workbench, `(name, steps, speed LoRA strength)`. Without them a clip renders at the speed LoRA's own settings.

`families/clips.py` holds the shared checks (`validate`, `read_frames`, `read_audio`, `grid_frames`). Your model package shouldn't need its own.

## 2. Caching

A clip's latent is `(C, T, h, w)`, and the shared cache stage encodes stills only. So a video driver implements `cache_stage(stage, datasets, args, device, aux)` and returns `True`:

- **latents**: decode each clip (the dataset item's frames are already checked against your spec), encode it with your video VAE, and store it. `families/cache.py`'s `save_latents(desc, item, latent, ...)` writes a `(C, T, h, w)` latent under a name the trainer finds; a sound track can ride beside it in `extra=`.
- **text**: the same as a stills model unless your conditioning differs per media kind.

Anything the cache stage needs that isn't a dataset setting arrives in `aux`, from your description's options as `aux:key=value` tokens. The sound decoder is the convention to follow: a model file with `role="audio_vae"` reaches the cache as `aux["audio_vae"]` and the driver as `options["audio_vae"]`.

H3's `cache_stage` hands off to its own caching scripts; read it for the full pattern, including voice-only items (a placeholder picture plus a sound latent).

## 3. Training

The trainer runs one item per step (batch size 1) and passes it straight through:

- `training_loss(dit, latents, cond, ...)` gets `(1, C, h, w)` for a photo and `(1, C, T, h, w)` for a clip. Tell them apart by `latents.dim()`.
- `batch_cond(batch, device)`: override it if your items carry more than the cached text (H3 maps its sound latent and an audio-only flag into `cond` here).
- `set_options(options)` runs before the dataset is built, so an option can shape it (H3's "clip still as photo" adds each clip's sharpest frame as a photo).

Per-media training uses two optional hooks the trainer calls every step:

| Hook | Use |
|---|---|
| `step_frozen_blocks(batch)` | Block ids whose weights sit out this item's step (H3 trains photos, clips and sound on different blocks) |
| `step_policy(batch, epoch)` | `(skip, lr_multiplier)`: skip a step entirely, or scale its learning rate (H3 retires a finished media category early) |

## 4. Previews and the workbench

**`generate(..., frames=, audio=)`** renders a clip when `frames > 1` and returns your own structure; H3 returns `{"latent", "audio", "size"}`. **`decode(vae, result, w, h)`** returns a PIL image for a still, or for a clip a dict: `{"frames": tensor [3, F, H, W] in 0–1, "image": the middle frame (PIL), "wave": sound [channels, L] or None}`. **`decode_audio(vae, audio)`** decodes the sound on its own, which the workbench needs when it decodes frames and sound separately. **`save_preview(result, path)`** writes a clip preview for the training gallery (H3 writes the frames, a `.wav` and an `.mp4`, then the middle-frame PNG last, because the gallery takes the PNG as the "finished" signal).

In the workbench, a family with clips gets the **generic video engine** (`families/video_workbench.py`) unless its description names its own. It renders and decodes clips through the four methods above. Repair Studio then plays clips side by side with sound, keeps a render library on disk, and shows the baseline clip beside the tweaked one. The clip render settings come from `clip_regimes`. Reuse unchanged blocks (`activation_cache=True`) replays the unchanged blocks on the first step, exactly as for stills.

The clip calls the tabs make (`render_clip`, `baseline_clip`, `nolora_clip`, `clip_from_cache`, `cache_key_for`) live in one shared class, `ClipContract`, which MiniMax H3's own engine also runs on. A family needs its own engine only for features beyond clips. H3's adds first/last-frame keyframes, reference pictures, a base-model picker and RefMod. It subclasses `ClipContract`, overrides the low-level render, and declares what it adds (`supports_keyframes`, `supports_base_modes`, `supports_banks`) so the tabs show those controls only for it.

## 5. Memory

Clips multiply memory by their frame count, so a video family usually needs its own plan:

- **`plan_run(...)`** (optional) is the trainer's Auto plan for precision and block swap. H3 plans from its largest clip's frames × pixels.
- **`clip_bucket_cap(free_gb, w, h)`** caps a clip's cached size to what your VAE can encode in the free VRAM, so a resolution that suits the photos doesn't fail on the clips. The default doesn't cap.
- **`park_for(...)` / `unpark(...)`**: clip decodes are heavy. Return a token after freeing room beside the training model for a preview decode (H3 parks only as many tail blocks as it needs).

## What's still H3-specific

Two tools are built around H3 today: RefMod Studio, and Gizmo's clip lengths (its spec is H3's). A new video family trains, previews (the training gallery plays whatever `save_preview` writes), and opens in Repair Studio, the Profiler, Extract and LoRA Royale without either.
