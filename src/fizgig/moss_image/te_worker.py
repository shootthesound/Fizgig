"""MOSS-Image text-encoder worker. Runs ONLY in the side venv (te_env.py: transformers >= 5.3) as a plain script -
it imports torch, transformers and safetensors, never Fizgig - and serves te_proxy.MossTextEncoder over stdin/stdout.

What it computes is the reference pipeline's MOSSImagePipeline.encode_prompt, per caption and unpadded:
    text = apply_chat_template([{"role": "user", "content": caption}], add_generation_prompt=True,
                               enable_thinking=True)   # == TEMPLATE.format(caption.strip()): the jinja trims
    ids = tokenizer(text, truncation=True, max_length=4096)
    hidden = Qwen3_5Model(ids, use_cache=False).last_hidden_state[0]
Only the language model is built (Qwen3_5TextModel; the vision tower and its 0.9 GB are never loaded): for text-only
input its positions equal the multimodal model's, which the tests check.

Protocol, one JSON object per line:
    -> {"id": 1, "captions": ["...", ...]}       <- {"id": 1, "path": "<tmp>/1.safetensors", "lens": [L0, ...]}
    -> {"id": 2, "quit": true}                   (exit)
    <- {"ready": true, "transformers": "5.12.1", "mode": "..."} once loaded; {"id", "error"} on a failed request.
The safetensors file holds h0..hN, each (L_i, 4096) in the output dtype. Logs go to stderr.

Modes: "full" (whole model on the device), "stream" (embeddings and norm resident, each decoder layer moved to the
device for its forward and back - exact, ~3 GB of VRAM; the default below ~19 GB free), "cpu".

Stream mode runs a request layer-major (--order layer, the default): all its captions go through layer 0, then all
through layer 1, ..., so each layer crosses to the GPU once per request instead of once per caption. Each caption
still runs alone and unpadded through the very same layer calls, so the result is bit-for-bit the per-caption path
(--order caption, kept for checks). Captions are grouped to --group_tokens tokens (the between-layer hidden states:
2 GB at the default); each group is one pass over the weights.
--weights: where the decoder layers live between passes - "resident" (read once into pinned RAM, ~17 GB for the
9B) or "disk" (read from the checkpoint at each pass: for machines without that much free RAM).
"""
import argparse
import json
import os
import sys
import tempfile
import time

TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
MAX_TOKENS = 4096
_PREFIXES = ("model.language_model.", "language_model.model.", "language_model.", "model.")


def log(msg):
    print(f"[moss te] {msg}", file=sys.stderr, flush=True)


_PROTOCOL = sys.stdout       # the proxy's channel; everything else printing to stdout (transformers' own checks)
sys.stdout = sys.stderr       # goes to stderr instead, so it can never corrupt a reply


def send(obj):
    _PROTOCOL.write(json.dumps(obj) + "\n")
    _PROTOCOL.flush()


def weight_files(path):
    if os.path.isdir(path):
        idx = [f for f in os.listdir(path) if f.endswith(".safetensors.index.json")]
        if idx:
            path = os.path.join(path, idx[0])
        else:
            return sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".safetensors"))
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        return [os.path.join(os.path.dirname(path), s) for s in shards]
    return [path]


def text_keys(files):
    """The language model's tensors only: {bare name (embed_tokens..., layers.N..., norm.weight): (file, key)}."""
    from safetensors import safe_open
    keys = {}
    for fn in files:
        with safe_open(fn, "pt") as f:
            for k in f.keys():
                if "visual." in k or k.startswith(("lm_head.", "mtp.")) or ".mtp." in k:
                    continue
                keys[next((k[len(p):] for p in _PREFIXES if k.startswith(p)), k)] = (fn, k)
    return keys


def read_tensors(keys, names):
    """{name: tensor} for `names`, one open per file."""
    from safetensors import safe_open
    by_file = {}
    for n in names:
        by_file.setdefault(keys[n][0], []).append(n)
    out = {}
    for fn, ns in by_file.items():
        with safe_open(fn, "pt") as f:
            for n in ns:
                out[n] = f.get_tensor(keys[n][1])
    return out


def text_state_dict(files):
    keys = text_keys(files)
    return read_tensors(keys, list(keys))


class LayerWeights:
    """One decoder layer's weights between passes: in RAM ("resident", pinned when the device is CUDA) or in the
    checkpoint ("disk", empty placeholders in between). load() puts them on the device, release() sends them home."""

    def __init__(self, layer, prefix, keys, dtype, device, home, pin):
        self.device, self.dtype, self.home = device, dtype, home
        import torch
        self.pin = pin
        if home == "disk":
            # meta parameters cannot take real data: plain empty placeholders, filled at each load()
            for mod in layer.modules():
                for n, t in list(mod.named_parameters(recurse=False)):
                    setattr(mod, n, torch.nn.Parameter(torch.empty(0, dtype=t.dtype), requires_grad=False))
        self.params = dict(layer.named_parameters())
        self._empty = {n: torch.empty(0, dtype=t.dtype) for n, t in self.params.items()}
        if home == "disk":
            self.names = {f"{prefix}{n}": n for n in self.params}
            self.keys = keys
            self.homes = None
        else:
            self.homes = {}
            for n, t in self.params.items():
                h = t.data.to("cpu")
                if self.pin[0]:
                    try:
                        h = h.pin_memory()
                    except RuntimeError:       # not enough page-locked RAM: plain copies, slower transfers
                        self.pin[0] = False
                        log("pinned memory unavailable - streaming from pageable RAM")
                t.data = h
                self.homes[n] = h

    def load(self):
        if self.homes is not None:
            for n, t in self.params.items():
                t.data = self.homes[n].to(self.device, non_blocking=True)
        else:
            got = read_tensors(self.keys, list(self.names))
            for full, n in self.names.items():
                self.params[n].data = got[full].to(self.device, self.dtype)

    def release(self):
        for n, t in self.params.items():
            t.data = self.homes[n] if self.homes is not None else self._empty[n]


def load_model(te_path, config_path, dtype, device, mode, weights="resident", order="layer"):
    """The text model. In stream mode it gets .layer_weights (one LayerWeights per decoder layer); with
    order "caption" each layer also fetches its weights around its own forward (hooks), as before."""
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    cfg = AutoConfig.from_pretrained(config_path)
    text_cfg = getattr(cfg, "text_config", cfg)
    with torch.device("meta"):
        model = Qwen3_5TextModel(text_cfg)
    keys = text_keys(weight_files(te_path))
    expected = set(model.state_dict())
    missing, unexpected = sorted(expected - set(keys)), sorted(set(keys) - expected)
    if missing or unexpected:
        raise KeyError(f"not the Qwen3.5 text encoder: missing {missing[:5]}, unexpected {unexpected[:5]}")
    stream = mode == "stream"
    from_disk = stream and weights == "disk"
    names = [k for k in keys if not (from_disk and k.startswith("layers."))]
    sd = read_tensors(keys, names)
    model.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=not from_disk, assign=True)
    del sd
    model.eval().requires_grad_(False)
    # the rotary table's inv_freq is a non-persistent buffer, left on meta: build that module for real
    model.rotary_emb = type(model.rotary_emb)(config=text_cfg)
    for name, buf in list(model.named_buffers()):
        if buf.is_meta:
            raise RuntimeError(f"buffer {name} still on meta after load")
    if not stream:
        model.to(device)
        model.layer_weights = None
        return model
    for name, child in model.named_children():
        if name != "layers":
            child.to(device)
    for buf_name, buf in model.named_buffers(recurse=False):
        setattr(model, buf_name, buf.to(device))
    pin = [str(device).startswith("cuda")]
    model.layer_weights = [LayerWeights(layer, f"layers.{i}.", keys, dtype, device, "disk" if from_disk else "resident",
                                        pin) for i, layer in enumerate(model.layers)]
    if order == "caption":
        for layer, w in zip(model.layers, model.layer_weights):
            layer.register_forward_pre_hook(lambda mod, args, kwargs, w=w: w.load(), with_kwargs=True)
            layer.register_forward_hook(lambda mod, args, kwargs, out, w=w: (w.release(), out)[1], with_kwargs=True)
    return model


class _Recorder:
    """Stands in for a decoder layer while the model's own forward runs: records the call (the mask and position
    tensors that layer would get) and passes the hidden states through untouched."""

    def __init__(self):
        self.hidden, self.kwargs = None, None

    def __call__(self, hidden_states, **kwargs):
        self.hidden, self.kwargs = hidden_states, kwargs
        return hidden_states


def layer_calls(model, ids):
    """The model's own forward with every decoder layer swapped for a recorder: the embedded input and, per layer,
    the keyword arguments Qwen3_5TextModel.forward hands it (its causal or linear-attention mask, rotary tables,
    positions). Cheap: no decoder layer runs."""
    import torch.nn as nn
    real = model.layers
    recs = [_Recorder() for _ in real]
    shim = nn.ModuleList([nn.Identity() for _ in real])       # (ModuleList wants modules; calls go to recs)
    for m, r in zip(shim, recs):
        m.forward = r.__call__
    model.layers = shim
    try:
        model(input_ids=ids, use_cache=False, return_dict=True)
    finally:
        model.layers = real
    return recs[0].hidden, [r.kwargs for r in recs]


def groups_by_tokens(lengths, budget):
    groups, cur, n = [], [], 0
    for i, L in enumerate(lengths):
        if cur and n + L > budget:
            groups.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += L
    if cur:
        groups.append(cur)
    return groups


def encode_layer_major(model, ids_list, device, out_dtype, group_tokens):
    """last_hidden_state[0] for each input, layer-major: per group, each layer's weights go to the device once and
    run over every caption of the group, one caption at a time (the same calls as the per-caption forward)."""
    import torch
    hidden_size = model.config.hidden_size
    out = [None] * len(ids_list)
    groups = groups_by_tokens([int(x.shape[1]) for x in ids_list], group_tokens)
    n_layers = len(model.layers)
    for gi, g in enumerate(groups):
        t0 = time.time()
        tokens = sum(int(ids_list[i].shape[1]) for i in g)
        store = "cpu"
        if str(device).startswith("cuda"):
            free = torch.cuda.mem_get_info(torch.device(device))[0]
            need = tokens * hidden_size * torch.finfo(model.norm.weight.dtype).bits // 8
            if need < min(3 << 30, free - (4 << 30)):
                store = device            # the group's hidden states fit beside the layer: they never leave the GPU
        hidden, calls = [], []
        for i in g:
            h, kw = layer_calls(model, ids_list[i].to(device))
            hidden.append(h.to(store))
            calls.append(kw)
        for j, layer in enumerate(model.layers):
            w = model.layer_weights[j]
            w.load()
            try:
                for k in range(len(g)):
                    hidden[k] = layer(hidden[k].to(device), **calls[k][j]).to(store)
            finally:
                w.release()
            if (j + 1) % 8 == 0 or j + 1 == n_layers:
                log(f"layer {j + 1}/{n_layers}: {len(g)} captions" + (f" (group {gi + 1}/{len(groups)})"
                                                                     if len(groups) > 1 else "")
                    + f", {time.time() - t0:.1f}s")
        for k, i in enumerate(g):
            out[i] = model.norm(hidden[k].to(device))[0].to("cpu", out_dtype).contiguous()
            hidden[k] = None
        del calls
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--te", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", default="full", choices=("full", "stream", "cpu"))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out_dtype", default="bfloat16")
    ap.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--weights", default="resident", choices=("resident", "disk"))
    ap.add_argument("--order", default="layer", choices=("layer", "caption"))
    ap.add_argument("--group_tokens", type=int, default=1 << 18)
    a = ap.parse_args()
    import torch
    import transformers
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    if (major, minor) < (5, 3):
        raise SystemExit(f"transformers {transformers.__version__} < 5.3 - run src/fizgig/moss_image/te_env.py with Fizgig's python")
    device = "cpu" if a.mode == "cpu" else a.device
    dtype, out_dtype = getattr(torch, a.dtype), getattr(torch, a.out_dtype)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    probe = "a red fox, {braces} and \\n ünïcode"
    rendered = tok.apply_chat_template([{"role": "user", "content": probe}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
    if rendered != TEMPLATE.format(probe.strip()):
        log(f"chat template differs from the expected string ({rendered!r}); using apply_chat_template")
    model = load_model(a.te, a.config, dtype, device, a.mode, a.weights, a.order)
    layer_major = a.mode == "stream" and a.order == "layer"
    how = f", {a.weights} weights, {a.order}-major" if a.mode == "stream" else ""
    log(f"loaded in {time.time() - t0:.1f}s ({a.mode}, {a.dtype}, {device}{how})")
    out_dir = tempfile.mkdtemp(prefix="moss_te_")
    send({"ready": True, "transformers": transformers.__version__, "mode": a.mode, "out_dir": out_dir,
          "weights": a.weights if a.mode == "stream" else None, "order": a.order if a.mode == "stream" else None})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("quit"):
            break
        rid = req.get("id")
        try:
            ids_list = []
            for cap in req["captions"]:
                text = tok.apply_chat_template([{"role": "user", "content": cap}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=True)
                ids_list.append(tok(text, return_tensors="pt", truncation=True, max_length=a.max_tokens).input_ids)
            with torch.inference_mode():
                if layer_major:
                    hs = encode_layer_major(model, ids_list, device, out_dtype, a.group_tokens)
                else:
                    hs = [model(input_ids=ids.to(device), use_cache=False, return_dict=True).last_hidden_state[0]
                          .to("cpu", out_dtype).contiguous() for ids in ids_list]
            tensors = {f"h{i}": h for i, h in enumerate(hs)}
            lens = [int(h.shape[0]) for h in hs]
            path = os.path.join(out_dir, f"{rid}.safetensors")
            save_file(tensors, path)
            send({"id": rid, "path": path, "lens": lens})
        except Exception as e:                       # noqa: BLE001 - reported to the proxy, the worker lives on
            import traceback
            traceback.print_exc(file=sys.stderr)
            send({"id": rid, "error": f"{type(e).__name__}: {e}"})
    log("bye")


if __name__ == "__main__":
    main()
