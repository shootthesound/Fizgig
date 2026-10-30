# Fizgig v8.9.93: themes, and pods on your computer's version

**Themes.** **Preferences → Theme** has eight looks for Fizgig: Fizgig (the dark blue you know), Light n' Orange, Seventies, Synthwave and Graphite, and three for eyes: **Colour-blind safe** (red-green colour blindness: good, bad and middling are blue, vermilion and yellow, never red against green), **High contrast** (low vision: black, white text, a yellow accent, underlined links) and **Dim** (light sensitivity or migraine: low glare, soft warm greys, no white). Each device keeps its own, so a phone can look different from the computer. Your pictures, masks and previews look the same in every theme. [Themes](https://github.com/shootthesound/Fizgig/blob/master/docs/WEB_UI.md#themes)

<p align="center"><img src="https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/assets/screenshots/theme_light_orange.png" alt="Light n' Orange" width="32%"> <img src="https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/assets/screenshots/theme_seventies.png" alt="Seventies" width="32%"> <img src="https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/assets/screenshots/theme_synthwave.png" alt="Synthwave" width="32%"></p>

**A pod runs the same Fizgig as your computer.** When your Fizgig opens a pod (in its app window, or a pod on its Pods tab), it puts the pod on your computer's version, so moving a run, helpers and the app window always work together. It takes a few seconds, and the pod keeps that version at every start. A busy pod (a run, a download, captions, a move) switches once it's free, a pod set to a release with `FIZGIG_REF` keeps its own, and the pod's **About** box has **Use the newest Fizgig**. Your computer needs this version for it, so update it first. [The same Fizgig as your computer](https://github.com/shootthesound/Fizgig/blob/master/docker/README.md#the-same-fizgig-as-your-computer)

**Prompt sliders: prompts to leave alone.** A prompt slider can now list prompts the dial must not change (Training tab, **Leave these alone**, with **How firmly**): a hair-length slider keeps a man's face the same, say. The previews show the first one beside the dial, and the run card edits them while it trains (thanks @driqeks, #252).

**MOSS-Image:** the **Fast** preset (rank 16, 0.25 MP) is now the default.

**The loading screen** shows a short tip each time Fizgig starts.

**Fixes**

- **Training on sub-folders trained Fizgig's own folders:** `originals_2`, `originals_3`… (Image Prep's Keep them safe on a later run) and `removed` (where Remove puts pictures) trained as if they were your pictures, and an uncaptioned one stopped Start. They're left out now, in any case, and the run's log names them (thanks @Murf-Code, #259).
- **The app window opened off screen** after closing Fizgig from the taskbar while it was minimized (Windows). It opens where it should now (thanks @Murf-Code, #259, likely #205).
- **Downloads failed once a Hugging Face login expired**, even public ones like SAM 3.1, with "Repository Not Found". Fizgig tries again without the login and says so in one line; a model that needs a login says plainly that it has expired (thanks @Murf-Code, #259).
- **A run that runs out of GPU memory** now ends with what to do: **Continue** on its card in Runs & outputs carries on from its newest save, with fresh memory (thanks @kiwiqq-jpg, #258).
- **An SDXL prompt slider with CFG above 1** stopped at its first practice picture.
- **The Look Consistency Filter scored masks as pictures** (`-masklabel.png`), found no face in most and suggested moving them. It leaves masks out now (thanks @Ferocit, #247).
- **Long jobs grey their buttons** while they run, the page says it's starting instead of looking stuck, and the cache stages count up on the run card (thanks @Ferocit, #249).
- **Masked training:** the row **Outside the mask, keep the model's own picture** says *(prior preservation)* in its title, and its help is one line per choice.

Train on sub-folders came in 8.9.92: [Fizgig v8.9.92 release notes](https://github.com/shootthesound/Fizgig/releases/tag/v8.9.92).

**Updating:** update Fizgig on your computers, and restart your pods.
