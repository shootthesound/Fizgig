# Fizgig with AI apps and Affinity (MCP)

Fizgig can be driven by an AI model through the Model Context Protocol (MCP), and it can trade pictures with Affinity by Canva. Three ways to use it:

- **The Assistant tab.** A language model running on your own computer does what you ask in plain words, using Fizgig's tools and, when it's open, Affinity. Nothing leaves your computer.
- **Another AI app** (Claude Desktop, LM Studio, Cursor ...) uses Fizgig as one of its MCP servers.
- **Affinity by Canva**, through its own MCP server: layers out to a training folder, renders back in as layers.

## Switch it on

**Preferences → Network → Let AI apps use Fizgig (MCP).** It is off until you tick it, it answers only apps on the computer running Fizgig, and a web page can never reach it. The Assistant tab works either way: it runs inside Fizgig.

Fizgig's MCP address is `http://127.0.0.1:8081/mcp` (your web UI's port). For an app that starts MCP servers itself over stdio, point it at:

```
python src/fizgig/scripts/fizgig_mcp.py
```

with Fizgig running. Claude Desktop's `claude_desktop_config.json`, for example:

```json
{"mcpServers": {"fizgig": {"command": "W:\\path\\to\\Fizgig\\venv\\Scripts\\python.exe",
                           "args": ["W:\\path\\to\\Fizgig\\src\\fizgig\\scripts\\fizgig_mcp.py"]}}}
```

## What the tools do

| Tool | What it does |
|:--|:--|
| `fizgig_status` | What's running: the training run, Studio, captioning, Image Prep |
| `list_models` | The models Fizgig trains, with their presets |
| `list_training_folders`, `create_training_folder`, `add_pictures`, `set_training_folder` | Training folders |
| `list_pictures`, `look_at_picture`, `write_caption`, `caption_folder` | Pictures and captions (AI captions wait until they're done) |
| `make_masks` | Masks for masked training |
| `start_training`, `training_status`, `wait_for_training`, `pause_training`, `stop_training` | Training runs. Start shows the Start checks' warnings first |
| `list_runs`, `list_loras`, `list_previews` | Results |
| `render` | Pictures with a LoRA, on Studio's preview model: several in one call load the model once. A LoRA at strength 0 renders with the plain model |
| `score_pictures` | Scores pictures of a person before one is used, no GPU needed: likeness to their training folder (ArcFace), eyes open, facing the camera, one main face, sharpness; picks the best likeness among those that pass. ONNX models on onnxruntime |
| `cloud_train`, `cloud_status`, `cloud_cancel` | Training on a rented RunPod GPU (your RunPod API key on the Pods tab): the best-value card under your hourly price, a pod that doesn't start in time deleted and the next tried, the LoRA files brought home and the pod deleted. Fizgig stops it at the spending cap or time limit you set |
| `release_gpu` | Takes Fizgig's models off the GPU for another program |
| `affinity_status`, `affinity_documents`, `affinity_view` | What's open in Affinity, and how it looks |
| `affinity_add_shape`, `affinity_add_text`, `affinity_hide_layer`, `affinity_save` | Lay out a design: rectangles (solid or gradient fill, outline), lines of text, hiding a layer (never deleting), saving |
| `affinity_to_training_folder` | Exports a document's layers (or the selected ones, or the whole document) into a training folder |
| `affinity_new_document` | A new document of a set size, saved in the Fizgig Bridge folder |
| `affinity_place` | Puts a picture into Affinity as a new layer, fitted into a layer's box if you name one; with replace, the old layer is hidden (never deleted) |

## Affinity by Canva

Affinity 3.2 and later has its own MCP server. In Affinity: **Edit → Settings → Model Context Protocol**, turn on **Enable Affinity MCP** and **Access files on your Desktop**, then quit and reopen Affinity.

Without any AI model: the Start tab's **From Affinity** card makes a training folder from a document's picture layers, and **Send to Affinity** on an enlarged picture (a preview, a Studio render) puts it into the open document as a new layer. Both show once Affinity answers.

Affinity's scripts can only reach the Desktop, so pictures cross through a folder there, **Fizgig Bridge**: `from Affinity` holds exported layers (Fizgig copies them into your training folder), `to Affinity` holds pictures on their way in. Fizgig never changes a document except by adding layers to it.

## The Assistant tab

Any model server with tool calling that speaks the OpenAI API works: llama.cpp's `llama-server` (start it with `--jinja`), LM Studio, Ollama. Put its address on the tab (for example `http://127.0.0.1:38088/v1`), press **Detect**, and pick the model. A model that can see pictures (a vision model) can look at renders and at your Affinity document.

**Free the GPU for Fizgig** (on by default): when the model and Fizgig's models don't fit on the card together, the model steps off the GPU before a render, captioning or training, and comes back after. That needs llama.cpp in router mode (`--models-preset`), which Fizgig can ask to unload and reload the model; other servers (LM Studio, Ollama, llama.cpp started with `-m`) can't be asked, so their model stays on the card beside Fizgig's. The line under the conversation shows how many handovers there were and what they cost. On a machine with memory for both (a large unified-memory system), untick it: nothing swaps. Router mode takes a presets file naming the model:

```ini
; presets.ini - the section name is the model's name in the Assistant tab
[my-model]
model = C:\models\Qwen3.8-27B-UD-Q4_K_XL.gguf
mmproj = C:\models\mmproj-F16.gguf
n-gpu-layers = all
ctx-size = 32768
```

```
llama-server --models-preset presets.ini --models-max 1 --jinja --port 38088
```


Things to try:

- *What's open in Affinity, and what is Fizgig doing?*
- *Make a training folder from the layers in my Affinity document, caption it with the trigger word gwelt, and train a quick likeness.*
- *Look at my poster in Affinity and make a portrait for its empty photo box that suits its colours.*
