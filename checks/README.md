# Checks

Quick, self-contained checks that run on every pull request (`.github/workflows/checks.yml`) and anyone can run at home. They catch shared code breaking for another model: a family description that no longer validates, sampler maths that drifted, a key mapping that lost a layer.

## Run them

```bash
pip install -r checks/requirements.txt     # pytest, plus CPU torch if you don't have torch already
python -m pytest checks -q
```

## Adding a check

A check belongs here when it is:

- **CPU only.** No GPU, no CUDA.
- **Self-contained.** No model downloads and no model files. Build tiny tensors or toy models in the check itself.
- **Free of personal paths and data.** Nothing from your own folders, datasets or prefs.
- **Quick.** The whole folder should finish in a few minutes on a laptop.
- **Deterministic.** Seed anything random. A check that sometimes fails blocks everyone's pull requests.

## Simulated GPUs

Two helpers check GPU behaviour without a GPU:

- `simgpu.py` runs small real models on a CPU that pretends to be a card. It checks block swap: where each weight is when it's used, and that nothing mixes card and CPU tensors.
- `memsim.py` runs a full-size training launch, as the page starts it (caching, training, previews), on fake tensors on a simulated card. It reports the card memory the run allocates and reserves, and runs out of memory where a real card of that size would. Model files are their headers only (`memsim_headers/`, a few KB each), so it needs no downloads. It needs a torch built with CUDA (no card); on the CPU-only torch its checks are skipped.

Name the file `test_<what>.py` and keep it next to the others. If your feature needs a GPU or real model files to prove, keep that test on your side and describe what you measured in your pull request instead.
