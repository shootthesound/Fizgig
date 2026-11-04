# Fizgig + Affinity + a local LLM: the demo kit

Six demos, each one sentence typed into Fizgig's **Assistant** tab. A language model running on your own computer (Qwen3.8-27B in llama.cpp) does the work through two MCP servers: Fizgig's, which trains LoRAs, captions, renders and scores pictures, and the one built into Affinity by Canva, for documents and layers. You put in 15–25 photos of yourself; the demos train a model of you and design with it.

Each demo has a launcher. It starts what's needed (the language model's server, Fizgig, Affinity's document for that demo), sends the demo's sentence to the Assistant and opens the Assistant tab in your browser, so you can watch every step.

## Two ways to run the demos

**The AMD GPU route (recommended).** All of the demos' AI work (the language model, training, rendering, captioning and scoring) runs on an **AMD Instinct MI300X (192 GB)** rented on RunPod, while Affinity runs on your own PC. Your PC needs no particular GPU. This is how we ran the demos end to end on AMD.

**On your own PC.** Everything runs locally. For now this needs an **NVIDIA GPU with 24 GB or more**. We expect an AMD Radeon AI PRO R9700 (32 GB) would work too, but we haven't had one to confirm the whole demo on, and we didn't want to send instructions we hadn't run end to end.

## What you need

Both routes:
- Windows 11, Python 3.10 or later, and git.
- **Affinity by Canva** 3.2 or later (free). The setup installs it with winget if it's missing.
- 15–25 photos of yourself.

The AMD GPU route:
- A **RunPod account** with an API key and some credit. The MI300X costs about $2.40 an hour while it runs: about 10 minutes to set it up, then a few minutes per demo, so about $3–4 for all of them.
- About 10 GB of free disk space.

On your own PC:
- An **NVIDIA GPU, 24 GB or more**. The demos were recorded on an RTX 5090 (32 GB).
- About 100 GB of free disk space: Fizgig's models (about 40 GB), the language model (17–35 GB, sized to your card) and llama.cpp.

## Set up (once)

```
git clone -b wip-tools https://github.com/shootthesound/Fizgig
```

**The AMD GPU route:** run **`Fizgig\demos\0 - Set up for the AMD GPU route.bat`**. It installs Fizgig, then:
1. Switches on Fizgig's MCP server (this computer only).
2. Checks Affinity: installs it if it's missing, then waits while you turn on its MCP server (in Affinity, **Edit → Settings → Model Context Protocol**: turn on **Enable Affinity MCP** and **Access files on your Desktop**, then quit and reopen Affinity).
3. Asks for your photos (it opens `demos\my photos`), your name and role for the posters, and a trigger word for you.
4. Asks for your RunPod API key (make one in RunPod's console under **Settings → API Keys**).

Then run **`A - Run the demos on a rented AMD GPU (MI300X)`**. It waits for an MI300X to be free (RunPod's stock comes and goes), rents it, and installs Fizgig, its models and the language model on it (about 10 minutes). When it says it's ready, run the demos below. When you're done, run **`B - Back to this computer (delete the AMD GPU)`**: it brings your LoRAs home to `demos\AMD LoRAs` and deletes the pod, which is billed until then.

**On your own PC (NVIDIA):** run **`Fizgig\demos\0 - Set up on this computer (NVIDIA).bat`**. It installs Fizgig, then:
1. Downloads Fizgig's models: Qwen Image 2.1 and the captioner.
2. Switches on Fizgig's MCP server (this computer only) and points the Assistant at the local language model.
3. Downloads llama.cpp (build b10666, pinned) and the Qwen3.8-27B language model at a size that fits your card, and writes its settings.
4. Checks Affinity, as above.
5. Asks for your photos, your name and role, and a trigger word, as above.

Run either setup again any time: finished steps are skipped.

## The demos

Run them in order. Demo 1 trains the LoRA the others use. Keep Affinity open.

| Launcher | The sentence asks for |
|:--|:--|
| `1 - Your photos in Affinity to a trained portrait` | Your photos are laid out as layers of an Affinity document. The Assistant makes a training folder from the layers, captions them, trains a LoRA of you, renders 4 portraits, scores them and puts the best into Affinity |
| `2 - A transparent sticker into the poster` | A robot sticker with real transparency, placed in the top-right corner of a keynote poster (the launcher makes the poster) |
| `3 - The model reads the poster and fills the photo box` | The Assistant looks at the poster, matches its colours, renders portraits of you at the photo box's shape, scores them and fits the best into the box |
| `4 - Four variations laid out in Affinity` | Four photos of you in different settings, laid out as a 2×2 grid in a new document, each the best of 4 by score; then one is swapped for another |
| `5 - Trained on a rented cloud GPU` (optional, own-PC setup only) | Trains on a rented RunPod GPU under an hourly price cap ($2), brings the LoRA home, deletes the pod, and fills a poster on your PC. It needs a RunPod API key (Fizgig's Pods tab) and costs about $0.25 |
| `6 - A poster designed from nothing` | The Assistant designs a whole poster in Affinity (background, type, photo frame), renders 4 portraits, scores them and places the best |

### Our timings

Each demo was measured from its launcher, sentence to finished document, on 3 November 2026:
- **This computer:** an NVIDIA RTX 5090 (32 GB), Ryzen 9 9950X3D and 128 GB of RAM.
- **The AMD GPU route:** an AMD Instinct MI300X (192 GB) rented on RunPod, with Affinity on the same computer as above.

| Demo | RTX 5090 (32 GB): model swaps | RTX 5090 (32 GB) | MI300X (192 GB) |
|:--|--:|--:|--:|
| 1. Photos → trained portrait | 3 | 6 min 33 s | 5 min 11 s |
| 2. Transparent sticker | 1 | 1 min 36 s | 55 s |
| 3. Fill the photo box | 1 | 2 min 8 s | 1 min 38 s |
| 4. Social pack with a swap | 5 | 7 min 0 s | 3 min 51 s |
| 5. Trained on a rented cloud GPU | – | 14 min 42 s ($0.12) | – |
| 6. Poster from nothing | 1 | 3 min 55 s | 3 min 15 s |

- **On the RTX 5090 (32 GB),** the language model and Fizgig's models can't share the card, so they swap each time the Assistant renders or trains. On the MI300X (192 GB) all of them stay loaded: no swaps at all, with 83.7 GB of the 192 GB in use at the peak.
- **The language model decides how to batch its work,** so the same demo varies by about ±20% between runs on either machine. Demo 1 on the MI300X (192 GB) took 5 min 11 s, 5 min 46 s and 6 min 16 s in three runs.
- **On the AMD GPU route, every picture crosses the internet** between your PC and the pod. Each crossing measured under a second.
- **Demo 5** rents its own cloud GPU and renders on your PC, so it's part of the own-PC setup only.

`Stop the demo servers` stops the language model and the demos' Fizgig (port 8095). Affinity stays open.

## More on the AMD GPU route

**`A - Run the demos on a rented AMD GPU (MI300X)`** rents an AMD Instinct MI300X (192 GB) on RunPod, about $2.40 an hour, with the RunPod key the setup saved.
- **Setup:** it waits for one to be free, then installs Fizgig, its models and the language model on it with AMD's ROCm image (about 10 minutes).
- **What runs where:** your Assistant then thinks, trains, captions, renders and scores on the AMD card, and Affinity stays on your PC. Pictures cross by themselves: a training folder made from your Affinity layers goes up, and a render comes down before it goes into Affinity.
- **The demos:** 1, 2, 3, 4 and 6, with their launchers.
- **Memory:** on the MI300X (192 GB), the language model, the render model and a training run all stay loaded together, so nothing swaps.

**`B - Back to this computer (delete the AMD GPU)`** brings every LoRA the pod trained home to `demos\AMD LoRAs`, deletes the pod and points the Assistant back at your PC. Run it when you're done: the pod is billed until then. If you set up with the own-PC setup as well, the demos then run locally again.

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
