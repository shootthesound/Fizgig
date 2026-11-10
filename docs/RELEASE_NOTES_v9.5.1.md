# Fizgig v9.5.1: Element boost, and new MOSS-Image settings

**Element boost (experimental), for MOSS-Image, in Training Parameters.** Early in a run, one part of each picture counts much more than the rest, so it's learned sooner. By default it's the face and hair, which brings likeness in faster, most of all when faces are small in the frame (medium and full-length shots). It runs three boosted stretches then one normal, so the body and background keep learning, for the first 12 epochs or 600 steps, whichever comes first; after that the run trains as normal. It works with or without masks (a mask trims it to the person), and a picture where nothing is found trains as normal. It's one tick: there's nothing to tune.

**What it boosts:** Face and hair (the default) or Face only, for a subject whose hair changes between pictures. The other choices are the ones Make masks offers (the person, the clothes, the body, the background and so on), found by the same person finder. The clothes choice suits an outfit LoRA rather than a person one.

**It's on in MOSS-Image's Fast, Standard and Turbo Train presets,** and off in Style, Slider and Fine-tune, where a single face isn't the point. Element boost is for MOSS-Image only for now.

**MOSS-Image's Clean-end share now defaults to `11:7:40:8:3`.** That's 11% of steps on nearly clean images until epoch 7 (about the model's own share), then 40% from epoch 8, with every 3rd epoch after that back at 11% at half the learning rate. Muon, every preset's optimizer, trains differently from AdamW, so the settings that worked best before aren't the sweet spot any more. Starting at 11 lets composition settle first. In our tests it also stopped the hair drifting darker late in a run, and likeness late in the run was higher. It was tuned together with Element boost: the boost does the early face work, and the lower clean-end share at the start leaves room for the rest of the picture. The Fine-tune preset keeps `11:4:30:8:2`. The card's "Click here to try them" puts the earlier `30:7:60:8:3` in the boxes. Load a ✨ preset to pick up the new values; saved settings keep what they had.

**Also in this release:**
- Training tab: related settings sit together in one box each (Save state, Gradients, Learning-rate schedule, Buckets, Metadata, Preview size, Turbo LoRA, and a fine-tune's Rotations and Regularisation). LR Scheduler and Warmup steps move from Other Options to the Optimizer section. The Metadata box fits about three to a line, one on a phone.
- Training maps: Learned since the start can also show each part's share of its own starting error, and that share on one scale for every run and epoch, so two runs compare by eye. Amount stays the default.
- Studio: a Primary + donor, 50/50 preset on every model.
- In the app window, F5 and Ctrl+R reload the page as in a browser, and if the previews gallery can't refresh it now says why instead of freezing at its last picture.
- AMD (ROCm): JoyCaption in 4-bit works on ROCm 7.15 (#316). Thanks @AlexWolfgangDRG for the diagnosis.
- The Fizgig 8 welcome no longer shows at start-up.
- Install guide: a new entry for training that's suddenly several times slower on Windows, which is usually part of it spilling into shared GPU memory, and how to stop it in the NVIDIA Control Panel.

**For command-line users:** Element boost is `--face_weight 9 --face_area head --face_cycle 3:1 --face_until 12:600`. `--face_area` takes `face`, `head`, or a Make masks choice (`person`, `no_clothes`, `body_clothes`, `body`, `clothes`, `not_faces`, `background`). See the [training guide](https://github.com/shootthesound/Fizgig/blob/master/docs/TRAINING.md#element-boost-one-part-of-each-picture-learns-faster).

**Updating:** run the updater as usual. Nothing new installs. The person finder Element boost uses for its non-face choices is the 16 MB model Make masks already uses; it downloads the first time if you've never made masks.
