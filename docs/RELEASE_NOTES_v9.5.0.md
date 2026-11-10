# Fizgig v9.5.0: JoyCaption and booru tags on the Captions tab

**Two new captioners on the Captions tab:**
- **JoyCaption Beta One:** an 8B model made for writing training captions. It's good with people (it tells men from women, and isn't fooled by mirrors and reflections) and with every style. It follows the same instructions as Qwen3-VL, and also offers JoyCaption's own caption types (Descriptive, Straightforward, Stable Diffusion prompt, MidJourney prompt, Art critic) with a Length, plus extra options to tick: lighting, camera angle, shot type and more. The box under the ticks takes your own instructions, and the line below shows exactly what it's told. It runs in full precision with about 19 GB free, otherwise in 4-bit, which needs about 8 GB. Thanks @mart-hill for asking (#313).
- **WD EVA02 tagger:** booru tags (1girl, long hair, smile…) for anime and illustration, which is what Anima and anime SDXL models were trained on. Choose how sure a tag must be to keep it, and list tags never to write. Thanks @axymeus for asking (#315).

**Preferences has a new Captioning models card.** It shows the main captioning model (Krea 2's text encoder), the optional bigger Qwen3-VL (8B), and the other captioning models, each with a Download button. Each one also downloads itself the first time you use it (JoyCaption is 16 GB, the tagger 1.3 GB). Download ahead to have it ready or to caption offline. No Hugging Face token is needed.

**Auto-recaption is unchanged.** During training it always uses Qwen3-VL (the Krea 2 text encoder, or Qwen3-VL 8B if you've set it in Preferences), whichever model the Captions tab uses. The new captioners never load inside a training run, so training memory and speed are exactly as before.

**Adding a captioner is now like adding a model:** a description and a small class in `src/fizgig/captioners/`, and Fizgig does the rest. Start with a post in [New models & fine-tuning](https://github.com/shootthesound/Fizgig/discussions/new?category=new-models-fine-tuning), then see [Adding a captioner](https://github.com/shootthesound/Fizgig/blob/master/docs/ADDING_A_CAPTIONER.md).

**Also in this release:**
- Make masks (with SAM 3.1), the Look Consistency Filter and Extract now free the caption model's memory first, as training and Studio already did, so a smaller card doesn't run out.
- Detail crops follow their picture's mask. A mask you edit, add or delete updates the crops, and a picture replaced by one too small for a crop no longer leaves its old crop training. Thanks @mabseyuk.
- Batch size above 1 on MiniMax H3: a clip or a voice trains on its own step again, instead of stopping the run. Thanks @mabseyuk.

**For command-line users:** `batch_caption.py` now takes `--captioner <name>` (`qwen3vl` with `--model_file`, `florence2_promptgen`, `joycaption_beta_one`, `wd_eva02_large_v3` …) instead of `--backend`, `--text_encoder` and the `--florence_*` flags. See the [CLI guide](https://github.com/shootthesound/Fizgig/blob/master/docs/CLI.md).

**Updating:** run the updater as usual. Nothing new installs, and nothing downloads until you pick a new captioner. Pods you use from your own Fizgig follow your computer's version when they're free.
