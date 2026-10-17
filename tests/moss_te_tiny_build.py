"""Run in the MOSS side venv (transformers >= 5.3): build a tiny random Qwen3_5Model laid out like the official
text_encoder/ folder (config.json + a sharded index, keys language_model.* / visual.*), and the in-process reference
`last_hidden_state` for each caption, the way MOSSImagePipeline.encode_prompt computes it.

    python moss_te_tiny_build.py <out_dir> <tokenizer_dir> <captions.json>
"""
import json
import os
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer, Qwen3_5Config

out, tok_dir, caps_file = sys.argv[1:4]
captions = json.load(open(caps_file, encoding="utf-8"))
torch.manual_seed(0)
cfg = Qwen3_5Config(
    text_config=dict(vocab_size=248320, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_key_head_dim=8,
                     linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4,
                     full_attention_interval=4, max_position_embeddings=8192,
                     rope_parameters={"mrope_interleaved": True, "mrope_section": [1, 1, 0],
                                      "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"}),
    vision_config=dict(depth=1, hidden_size=32, intermediate_size=48, num_heads=2, out_hidden_size=64,
                       num_position_embeddings=64))
model = AutoModel.from_config(cfg).eval()
with torch.no_grad():               # visible values everywhere (HF init leaves some params at exactly 0 / 1)
    for n, p in model.named_parameters():
        p.add_(0.02 * torch.randn_like(p))
te = os.path.join(out, "text_encoder")
os.makedirs(te, exist_ok=True)
cfg.save_pretrained(te)
sd = {k: v.contiguous() for k, v in model.state_dict().items()}
keys = sorted(sd)
half = len(keys) // 2
shards = {"model-00001-of-00002.safetensors": keys[:half], "model-00002-of-00002.safetensors": keys[half:]}
wm = {}
for fn, ks in shards.items():
    save_file({k: sd[k] for k in ks}, os.path.join(te, fn))
    wm.update({k: fn for k in ks})
json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(te, "model.safetensors.index.json"), "w"))
tok = AutoTokenizer.from_pretrained(tok_dir)
ref, texts = {}, []
for i, c in enumerate(captions):
    text = tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True)
    texts.append(text)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=4096)
    with torch.no_grad():
        ref[f"h{i}"] = model(**inputs, use_cache=False, return_dict=True).last_hidden_state[0].contiguous()
save_file(ref, os.path.join(out, "reference.safetensors"))
json.dump(texts, open(os.path.join(out, "texts.json"), "w"))
print("ok")
