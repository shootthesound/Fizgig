# Fizgig + Affinity + a local LLM: the demo kit

Six demos, each one sentence typed into Fizgig's **Assistant** tab. A language model running on your own computer (Qwen3.8-27B in llama.cpp) does the work through two MCP servers: Fizgig's, which trains LoRAs, captions, renders and scores pictures, and the one built into Affinity by Canva, for documents and layers. You put in 15–25 photos of yourself; the demos train a model of you and design with it.

Each demo has a launcher. It starts what's needed (the language model's server, Fizgig, Affinity's document for that demo), sends the demo's sentence to the Assistant and opens the Assistant tab in your browser, so you can watch every step.

## What you need

- Windows 11.
- An **NVIDIA GPU, 24 GB or more**, for training and rendering. The demos were recorded on an RTX 5090 (32 GB). In this kit, training and rendering run on NVIDIA only. The language model also runs elsewhere: on any other GPU, the setup uses llama.cpp's Vulkan build.
- About 100 GB of free disk space: Fizgig's models (about 40 GB), the language model (17–35 GB, sized to your card) and llama.cpp.
- Python 3.10 or later, and git.
- **Affinity by Canva** 3.2 or later (free). The setup installs it with winget if it's missing.

## Set up (once)

```
git clone -b wip-tools https://github.com/shootthesound/Fizgig
```

Then run **`Fizgig\demos\0 - Set up everything.bat`**. It installs Fizgig, then:

1. Downloads Fizgig's models: Qwen Image 2.1 and the captioner.
2. Switches on Fizgig's MCP server (this computer only) and points the Assistant at the local language model.
3. Downloads llama.cpp (build b10666, pinned) and the Qwen3.8-27B language model at a size that fits your card, and writes its settings.
4. Checks Affinity: installs it if it's missing, then waits while you turn on its MCP server (in Affinity, **Edit → Settings → Model Context Protocol**: turn on **Enable Affinity MCP** and **Access files on your Desktop**, then quit and reopen Affinity).
5. Asks for your photos (it opens `demos\my photos`), your name and role for the posters, and a trigger word for you.

Run it again any time: finished steps are skipped.

## The demos

Run them in order. Demo 1 trains the LoRA the others use. Keep Affinity open.

| Launcher | The sentence asks for | Our time (RTX 5090) |
|:--|:--|--:|
| `1 - Your photos in Affinity to a trained portrait` | Your photos are laid out as layers of an Affinity document. The Assistant makes a training folder from the layers, captions them, trains a LoRA of you, renders 4 portraits, scores them and puts the best into Affinity | 6 min 33 s |
| `2 - A transparent sticker into the poster` | A robot sticker with real transparency, placed in the top-right corner of a keynote poster (the launcher makes the poster) | 1 min 36 s |
| `3 - The model reads the poster and fills the photo box` | The Assistant looks at the poster, matches its colours, renders portraits of you at the photo box's shape, scores them and fits the best into the box | 2 min 8 s |
| `4 - Four variations laid out in Affinity` | Four photos of you in different settings, laid out as a 2×2 grid in a new document; each the best of 4 by score; then one is swapped for another | 7 min |
| `5 - Trained on a rented cloud GPU` (optional) | Trains on a rented RunPod GPU under an hourly price cap ($2), brings the LoRA home, deletes the pod, and fills a poster here. It needs a RunPod API key (Fizgig's Pods tab) and costs about $0.25 | about 15 min |
| `6 - A poster designed from nothing` | The Assistant designs a whole poster in Affinity (background, type, photo frame), renders 4 portraits, scores them and places the best | 3 min 55 s |

`Stop the demo servers` stops the language model and the demos' Fizgig (port 8095). Affinity stays open.

Where things go:
- Affinity documents: the Desktop's `Fizgig Bridge` folder.
- Training folders and LoRAs: Fizgig's usual folders.
- Each demo's full log: `demos\logs`.

## What a 32 GB card costs, and what unified memory changes

The language model (about 22 GB loaded) and Fizgig's render model (about 20 GB) can't both stay on a 32 GB card, so they swap places each time the Assistant needs a picture. Fizgig times every swap and the log shows them:
- A render with the model already loaded takes about 4 s.
- A render that loads the model first takes about 40 s.
- Bringing the language model back takes 10–20 s.
- During training, the run holds the whole card and the Assistant waits.

On a machine with 96–128 GB of memory shared between CPU and GPU, the language model, the render model, the captioner and a training run all fit at once. The swapping and the waits go away.

## Picking the best picture: work for an NPU

`score_pictures` scores each render before it's used:
- likeness to your training photos (ArcFace);
- eyes open (from 106 face landmarks);
- facing the camera (head pose);
- one main face;
- sharpness.

The Assistant then places the best likeness among the pictures that pass. These are small ONNX models run by onnxruntime, about 0.2 s a picture on the CPU here. That is the kind of steady, low-power work an NPU is built for. On a Ryzen AI machine with its NPU provider (`VitisAIExecutionProvider`) installed, Fizgig picks that provider first. The models may need quantising for the NPU; we haven't run this on one yet. The result records where the scoring ran (`ran_on`) and how long it took.
