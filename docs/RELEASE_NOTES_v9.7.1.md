# Fizgig v9.7.1: every job keeps its own view

**Switch between the run that's training, the runs in your queue and what you're setting up, without losing your place in any of them.** A pill at the top right (bottom right on a phone) always says what the tabs are showing: **Viewing:** a job's name and what it is doing, or **unsaved run** when the form is no job yet. Press it and pick another: the Start, Image Prep, Captions, Samples and Training tabs all change to that job's training folder, captions, masks, sample prompts, settings and previews, and the view stays with you from tab to tab.

- **Training now** shows the run that's going. Its training settings are there to look at; its Samples tab stays changeable, and a change goes to its next previews.
- **Queued 1, 2, …** shows a queued run, and you can change it: **Save to queue** keeps the changes. A run you add with **Add to queue** becomes the job on show straight away.
- **A paused run is still that job.** The pill reads **Paused**, and everything Resume takes from the form (the previews, the size, block swap, the save schedule, the learning rate) is changed on its own settings.
- **Set up a new run with these settings** (in the pill's list on the Training tab) copies the job on show into a new setup with the same training folder, named with `_2` after its name, so another run of the same thing takes one press.
- **Back to the normal view** leaves the job for a clean start: no training folder picked, the model's default settings, no LoRA name. Your sample prompts stay.
- **When the job on show finishes,** the pill says so and pulses. The tab you're on keeps showing it, the list offers **Switch this tab to** the run that started next, and your next change of tab moves on by itself. The pulse colour is part of each theme.

**Likeness and Style scores belong to the run.** Pick your three reference photos once and they're remembered with the training folder, so every run on those pictures uses them without asking again. Each run keeps its own likeness and Style scores over its own previews, so two runs sharing a samples folder never mix or overwrite each other's numbers, and looking at an older run never clears the current one's. Scores made before this release are scored again the first time you open a run, and you pick the three photos once more per training folder.

**Star the previews you like.** A star on any preview in the Samples gallery, its grid or a run's card, and the matching epoch's checkpoint gets a star in the checkpoint lists, so the one you liked is easy to find when you pick a LoRA.

**A run that was cut off is offered again.** If Fizgig closes while a run trains, the next start asks what to do before anything runs: carry that run on from its last saved state (the queue follows after it), start the next run in the queue, or clear the queue and start fresh. The question you get when quitting now says where the run is saved, what would be lost, and what each button does.

**One name, one run.** Start, Add to queue and Check settings now refuse a LoRA name that already has files in the same output folder, and tell you which name is free. Two runs sharing a name wrote over each other's files, and deleting one took the other's. To train a run further, use **Continue** on Runs & outputs. **Start again with these settings** now names the new run with `_2` for the same reason.

**Captions tab: boxes that fit their caption.** Caption boxes are as tall as their text and grow as you type, in the cards, the table and the big view. The big view puts the caption beside the picture on a wide window, and under it otherwise. Suggested on Reddit, as was keeping each job's context.

**MiniMax H3:**
- **Ostris's training adapters move to v3.** The updater switches Preferences over from v1 or v2 and leaves the old file on disk; an adapter you chose by hand is kept. The Training adapter row is gone: the adapter is always on, and the right file is picked for your Training Base.
- **The Circlestone adapter is removed.** Ostris's v3 adapters did better in our tests. Its file is left on disk.
- **H3 Image edit LoRAs cache on 24 GB cards.** Caching the before-pictures crashed when the text encoder was streamed from system RAM (#334, thanks @DaRoamer for the diagnosis).
- **Studio frees the card properly before a text-encoder pass,** so a 24 GB card no longer runs out of memory there.

**Fixes:**
- **Batch size above 1 with block swap no longer crashes** with "mat2 is on cpu" at the first step (#335, thanks @spaghettified26 for a thorough report). A batch with block swap is slower per step than one without, because each picture moves blocks both ways.
- **Continue no longer picks up an older run's saved state** when a name was reused, which made a 20-epoch run read as 50 done (#333, thanks @AlexWolfgangDRG).
- **Delete run** also removes that run's likeness and Style score files.

**Also in this release:**
- **Model downloads say how they're going:** speed and time left beside the size, and the run's log names each file as it starts (#327, thanks @Ferocit).
- **Steps left in the epoch** on the run card, beside Step (#328, thanks @jordanjay29).
- **A line above Start when the Resume Training box holds a state,** saying which state Start would carry on from, with a button to clear it (#317, thanks @CognitiveDiffusion).
- **Preferences > App window: Draw the window without the graphics card,** so the window holds none of the VRAM a tight run needs (#321, thanks @daxter93). We couldn't test it on an AMD card, so tell us how it goes.
- **MOSS-Image's SFT look LoRA is for Fizgig's previews, not for ComfyUI:** its hint and the guide now say so. In ComfyUI, run MOSS-Image SFT itself.

**To update:** run the updater as usual. The first time you open a run's previews afterwards, pick the three likeness photos for that training folder once more; every run on those pictures then uses them.
