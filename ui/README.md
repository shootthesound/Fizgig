# Fizgig Web UI

A modern, responsive Next.js 15 Web interface and workbench for [Fizgig LoRA Studio](https://github.com/shootthesound/Fizgig).

---

## Overview

The Web UI brings Fizgig's complete LoRA workbench into the browser — enabling headless server training, remote browser access, and clean local usage with zero changes to Fizgig's core Python engines (`src/`).

### Features

- **Full Parity with Desktop Workbench**:
  - **Start**: Workflow overview, folder configuration, setup warnings.
  - **Training**: Complete parameter control for **Flux 2 Klein 9B**, **Krea 2**, and **MiniMax H3** with presets, live log streaming, and pause/resume.
  - **Captions**: Trigger word insertion, Florence-2/Qwen3-VL captioning, find & replace, image preview grid.
  - **Image Prep**: Multi-mode image preparation (Auto Prep, Resize, Face Crop) and target megapixel selection.
  - **Sample Previews**: Multi-prompt sample generation, resolution matrices, generation frequencies, flow shift, and CFG controls.
  - **Repair Studio**: Per-block transformer surgery sliders with interactive side-by-side previews and donor LoRA blending.
  - **LoRA the Explorer**: 4-variant evolutionary mutation matrix with seed drift and baseline promotion.
  - **LoRA Royale**: Epoch tournament, crossfade comparison, and ArcFace likeness rating.
  - **Profiler**: 5-tier block activation heatmaps.
  - **Extract**: SVD rank reduction and specialization.
  - **Metadata & Preferences**: Rich safetensors metadata inspector and model path configuration.
- **Design System Fidelity**:
  - Styled with Tailwind CSS configured directly from Fizgig's canonical Tkinter `COLORS` palette (`bg_deep`, `bg_surface`, `accent`, etc.).
- **Non-Destructive Core**:
  - Operates via REST API route handlers (`ui/src/app/api/`) that build and spawn headless CLI commands (`src/fizgig/scripts/train.py`, etc.) as documented in `docs/CLI.md`.
  - The native Tkinter desktop interface and existing CLI workflows remain 100% untouched.

---

## Getting Started

### Prerequisites

- Node.js 18.18+ (tested on Node.js 20 and Node.js 23)
- Python 3.10+ with standard virtual environment (`venv`)

### Installation & Launch

```bash
# Navigate to the Web UI folder
cd ui

# Install dependencies
npm install

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Production Build

```bash
cd ui
npm run build
npm start
```

---

## Architecture & API Bridge

The Web UI communicates with Fizgig's Python backend through dedicated Next.js Route Handlers:

| API Route | Purpose | Backend Script / File |
| :--- | :--- | :--- |
| `/api/training` | Starts, pauses, resumes, and monitors training runs | `src/fizgig/scripts/{train,krea2_train,minimax_train}.py` |
| `/api/datasets` | Generates and validates dataset TOML configs | `dataset/Fizgig_train.toml` |
| `/api/captions` | Caption generation, find & replace, and tagging | `src/fizgig/captioning/` |
| `/api/convert` | Image conversion, face-cropping, and resizing | `src/fizgig/image_prep/` |
| `/api/repair` | Block weight manipulation & companion sidecars | `src/fizgig/repair_studio/` |
| `/api/royale` | Multi-epoch comparisons and tournament scoring | `src/fizgig/royale/` |
| `/api/explorer` | Evolutionary block mutations | `src/fizgig/explorer/` |
| `/api/extract` | SVD rank extraction CLI builder | `src/fizgig/scripts/extract_lora.py` |
| `/api/profiler` | Per-block activation profile analysis | `src/fizgig/profiler/` |
| `/api/metadata` | Reads and writes SafeTensors ModelSpec headers | `safetensors` |
| `/api/gpu` | Host memory and GPU telemetry | `torch.cuda` / `os.sysinfo` |
| `/api/prefs` | Persists user settings and model directories | `prefs.json` |

---

## Automated Verification

The test suite validates backend script discovery, CLI schema matching, dataset TOML formatting, and end-to-end training lifecycles:

```bash
# Run tests from the repository root
python3 tests/test_env.py
python3 tests/test_dataset_toml.py
python3 tests/test_training_cli.py
python3 tests/test_backend_api_connections.py
python3 tests/test_full_training_flow.py
```
