# Fizgig v9.8.7: LoRA Isolation without split backgrounds, faster Krea 2 block swap, and Speed-ups for your graphics card

**LoRA Isolation: two subjects in one picture no longer split the background.** The side-by-side pictures LoRA Isolation adds are a photo of your subject next to a stranger, two backgrounds with a seam down the middle, and a LoRA could learn that as what two people together look like. The stranger's half of each one is now masked, so their background doesn't count in training (whether or not you train masked), and those pictures' captions end in words a prompt with your subjects doesn't use. In testing on Anima with two characters, no picture of both split, and each kept her own eyes and clothes better. Your subject's half trains as before, so there's nothing to set up; side-by-sides rebuild on your next run, and the strangers set downloads again with its masks. ([#407](https://github.com/shootthesound/Fizgig/issues/407), thanks @pomoe136)

**Krea 2: Faster block swap (send-only).** For LoRA training with Blocks Swap, each swapped block can be copied to the card and never copied back, at the same VRAM and with the same results. On an RTX 5090 (INT8, 16 blocks swapped) a step went from 3.28 to 2.17 s; a 16 GB laptop RTX 4090 ran at about the same speed with 0.7 GB less VRAM. It's off by default: tick it under Preferences → Speed-ups for your graphics card and compare s/step in the log. ([#403](https://github.com/shootthesound/Fizgig/pull/403), thanks @knoic, and @Murf-Code for testing)

**Preferences: Speed-ups for your graphics card.** The settings that help on some cards and not others are in one card, showing only the ones for your kind of card. NVIDIA: INT8 fast inference, Faster block swap, Keep swapped blocks in pinned RAM (off by default: 1.5x on an RTX 5090 with 128 GB of RAM, a little slower on some other machines), and on Windows a Show how for stopping the driver moving training into system RAM. AMD: INT8 fast inference, Force SageAttention, Split large convolutions and Use AMD's MIOpen.

**AMD: far less memory for previews and caching.** The Qwen-Image VAE (Anima, Krea 2) decodes in tiles, so a 1024x1536 preview no longer reserves about 30 GB on a 16 GB card and freezes Windows. Every model's large convolutions run in strips, about a third of the memory for the same result, and caching hands memory back after each picture instead of growing with each new picture size. ([#404](https://github.com/shootthesound/Fizgig/issues/404), thanks @AlexWolfgangDRG; [#389](https://github.com/shootthesound/Fizgig/issues/389), thanks @dkspwndj)

**Block swap no longer slows down as training goes on.** The speed LoRA and MOSS-Image's look LoRA wait on the CPU between previews, but some steps moved them onto the card and left them there, so training got slower and slower and could run out of memory. They now stay parked. ([#405](https://github.com/shootthesound/Fizgig/issues/405), thanks @spinolover)

**When the app window can't start, Fizgig says why.** If Windows' WebView2 runtime is missing or broken, Fizgig opens in your browser at once (it used to wait about 50 seconds) and the log says to reinstall WebView2. Closing that browser now closes Fizgig a few minutes later; a run that's training finishes first. ([#406](https://github.com/shootthesound/Fizgig/issues/406), thanks @AlexWolfgangDRG)

**Drag and drop.** Drag a file from your desktop onto any box with an Upload button: the training form's picture rows, the Samples reference pictures, Studio, LoRA Royale, the Profiler, Extract and Metadata. Each box takes only its kind of file. ([#392](https://github.com/shootthesound/Fizgig/issues/392), thanks @AlonDan)

**Captions search shows its matches.** The words you search for are highlighted in each caption. ([#400](https://github.com/shootthesound/Fizgig/pull/400), thanks @spaghettified26)

**Samples gallery likeness scoring.** The scorer says what it's doing, cancels cleanly, doesn't start counting again by itself, and leaves CPU cores free while it works. ([#382](https://github.com/shootthesound/Fizgig/pull/382), thanks @DmitryAkDev)

**LoRA Royale's Every epoch sheet takes the epochs you list.** Epochs typed in the Epochs box ("5,11,22,38") come straight from the training folder, without rendering them at the top first. ([#401](https://github.com/shootthesound/Fizgig/issues/401), thanks @axymeus)

**Regularisation sets train every kind of picture.** With a per-epoch limit, a picture size with less than one picture's share each epoch never trained; now it takes its fair share of the epochs. Thanks @mabseyuk.

**The Style score reads replaced pictures again.** A training picture replaced under the same name kept its old style reading; now it's read again. Thanks @mabseyuk.

**To update:** run the updater as usual.
