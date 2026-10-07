# Driver excerpts: Qwen Image 2.1 and MiniMax H3

Real code from two shipped drivers, at the places new drivers most often go wrong. Qwen Image 2.1 (`qwen_image21/driver.py`) is the reference for a stills model; MiniMax H3 (`minimax/driver.py`) for a model that trains on clips and sound. Read them beside [STILLS.md](STILLS.md) and [VIDEO.md](VIDEO.md).

Every block below is copied from the source by `src/fizgig/families/doc_excerpts.py`, and `--check` fails if one no longer matches, so what you read here is what Fizgig runs.

## Qwen Image 2.1: a stills driver

### Loading the text encoder for the card it's on

The encoder loads in bf16 when it fits beside room to run, else INT8, so caching works on a 10 GB card. Its tokenizer and processor load cache first inside `Qwen21TextEncoder`, the same rule `FamilyDriver.from_pretrained` gives every driver.

<!-- excerpt: qwen_image21/driver.py QwenImage21Driver.load_text_encoder -->
```python
def load_text_encoder(self, path, device):
    """Text-only (no LM head or vision tower): bf16 when it fits the free VRAM with room to run, else INT8 -
    about 8 GB, which fits a 10 GB card."""
    from fizgig.qwen_image21.embedder import Qwen21TextEncoder
    from fizgig.families.quant import free_vram_gb
    return Qwen21TextEncoder(path, device=device, int8=free_vram_gb() < 19.5)
```

<!-- excerpt: families/driver.py FamilyDriver.from_pretrained -->
```python
@staticmethod
def from_pretrained(cls, repo_id: str, **kwargs):
    """`cls.from_pretrained(repo_id, **kwargs)` from the local cache, the Hub only when nothing is cached. Plain
    from_pretrained asks the Hub on every call, so a fully cached tokenizer still fails offline or when the Hub
    rate-limits the machine. List the repo and files in the description's `helper_files` so the model
    downloader fetches them up front."""
    from fizgig.utils.hf_cache import from_pretrained_cache_first
    return from_pretrained_cache_first(cls, repo_id, **kwargs)
```

### Encoding: latents and conditioning on CPU, in the training space

`encode_images` gets uint8 arrays of one size and returns normalised latents on CPU; `encode_text` returns one dict per caption. The keys are the driver's own. The cache stores the dict and hands it back, batched, to `training_loss` and `generate`.

<!-- excerpt: qwen_image21/driver.py QwenImage21Driver.encode_images -->
```python
@torch.no_grad()
def encode_images(self, vae, images):
    x = torch.stack([torch.from_numpy(np.ascontiguousarray(a[..., :3])) for a in images])
    x = x.permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
    p = next(vae.parameters())
    return [z.to(torch.bfloat16).cpu() for z in S.encode_image(vae, x.to(p.device, p.dtype))]
```

<!-- excerpt: qwen_image21/driver.py QwenImage21Driver.encode_text -->
```python
@torch.no_grad()
def encode_text(self, te, captions):
    return [{"hidden_states": h.to(torch.bfloat16).cpu()} for h in te.encode(captions)]
```

### The training step

The model's own timestep distribution (logit-normal with the checkpoint's resolution-dependent shift), scaled into `[min_t, max_t]` so the Timestep Range control works. Flow matching: `x_t = (1 - t) x0 + t noise`, target `noise - x0`. It returns `{"t": t}` on the 0–1 scale where 0 is clean, which the per-image loss watch buckets by. The `diff_ref` branch is the slider pair weighting; `refs` the edit references.

<!-- excerpt: qwen_image21/driver.py QwenImage21Driver._sample_t -->
```python
@staticmethod
def _sample_t(n_tokens, generator, min_t, max_t):
    t = torch.sigmoid(torch.randn(1, generator=generator)).item()
    mu = S.calculate_mu(n_tokens)
    t = math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))
    return min_t + (max_t - min_t) * t
```

<!-- excerpt: qwen_image21/driver.py QwenImage21Driver.training_loss -->
```python
def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                  diff_weight=0.0):
    device = latents.device
    h, w = latents.shape[-2:]
    n = h * w
    x0 = S.pack(latents.float())
    noise = torch.randn(x0.shape, generator=generator).to(device)
    t = self._sample_t(n, generator, min_t, max_t)
    xt = (1 - t) * x0 + t * noise
    ref_mask = cond["ref_mask"][0] if "ref_mask" in cond else None
    enc, img_mask, enc_mask = S.model_inputs(cond["hidden_states"][0], n, device, ref_mask=ref_mask)
    shapes = [(1, h, w)]
    if refs:
        ref_tokens, ref_shapes = self._ref_sequence(refs, ref_mask, device)
        xt, shapes = torch.cat([ref_tokens, xt], dim=1), ref_shapes + shapes
    elif ref_mask is not None and ref_mask.any():
        raise RuntimeError("edit conditioning without its reference latents - re-cache the latents")
    tt = torch.tensor([t], device=device, dtype=torch.bfloat16)
    pred = dit(xt.to(torch.bfloat16), enc.to(torch.bfloat16), tt, [shapes], img_mask, enc_mask)[:, -n:]
    if diff_ref is not None and diff_weight > 0.0:
        # slider disentanglement: tokens where the two poles differ count more (Krea 2's formula)
        d = S.pack((latents.float() - diff_ref.to(device).float()).abs()).mean(dim=-1)     # (1, N)
        dm = d.mean(dim=1, keepdim=True)
        r = (d / dm.clamp_min(1e-8)).clamp(max=8.0)
        w = (1.0 - float(diff_weight)) + float(diff_weight) * r
        w = w / w.mean(dim=1, keepdim=True).clamp_min(1e-8)
        w = torch.where(dm > 1e-6, w, torch.ones_like(w))      # identical pair: uniform, never all-zero
        se = (pred.float() - (noise - x0)).pow(2).mean(dim=-1)
        return (se * w).mean(), {"t": t}
    return F.mse_loss(pred.float(), noise - x0), {"t": t}
```

### Sampling

An explicit sigma schedule (a speed LoRA's) applies only at its own step count, and the negative only above CFG 1. `on_step` is passed through to the sampler, which calls it before every step: that is how a render is cancelled and how Turbo Preview knows the step.

<!-- excerpt: qwen_image21/driver.py QwenImage21Driver.generate -->
```python
@torch.no_grad()
def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
             noise=None, on_step=None, refs=None):
    device = next(dit.parameters()).device
    neg = neg_cond["hidden_states"] if (neg_cond is not None and cfg > 1.0) else None
    neg_mask = neg_cond.get("mask") if neg is not None else None
    opts = dict(options)
    shift_terminal = opts.get("shift_terminal", S.SHIFT_TERMINAL)
    if sigmas is not None and len(sigmas) != steps:
        sigmas = None                   # an explicit schedule only applies at its own step count
    return S.sample(dit, cond["hidden_states"], height, width, steps=steps, seed=seed, cfg=cfg, neg_emb=neg,
                    device=device, sigmas=sigmas, shift_terminal=shift_terminal, noise=noise, on_step=on_step,
                    text_mask=cond.get("mask"), neg_mask=neg_mask, ref_latents=refs or None,
                    ref_mask=cond.get("ref_mask"), neg_ref_mask=neg_cond.get("ref_mask") if neg is not None else None)
```

### Prompt travel

LoRA Royale blends waypoint prompts, so they must share one shape. Zero padding plus a mask of the real tokens, and a model that masks padded keys, makes each waypoint render exactly as its own prompt.

<!-- excerpt: qwen_image21/driver.py QwenImage21Driver.pad_conditioning -->
```python
def pad_conditioning(self, conds):
    """Pad prompts to one length (zeros) with a mask so they can be blended (prompt travel). The DiT masks the
    padded keys and starts the image's positions after the real tokens, so a padded prompt renders as the
    unpadded one does."""
    L = max(c["hidden_states"].shape[0] for c in conds)
    out = []
    for c in conds:
        h = c["hidden_states"]
        pad = L - h.shape[0]
        out.append({"hidden_states": torch.cat([h, h.new_zeros(pad, h.shape[1])]) if pad else h,
                    "mask": torch.cat([torch.ones(h.shape[0], dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)])})
    return out
```

### Fine-tune: a fused weight in the file

ComfyUI's file fuses the MLP's gate and projection into one `gate_up` tensor while the model splits it; `file_layout` maps each Linear to its rows, so the master reads that slice and checkpoints are written back in the file's own layout. The memory figures are measured window peaks (see the [checklist](CHECKLIST.md#small-cards)).

<!-- excerpt: qwen_image21/driver.py QwenImage21Driver.ft_spec -->
```python
def ft_spec(self, dit):
    # four balanced windows per block (attention ~0.13 GB, each MLP matrix ~0.10); the file fuses the MLP's gate
    # and projection into one gate_up tensor, [gate_layer; proj]
    from fizgig.families.ft import FTSpec
    return FTSpec(blocks="transformer_blocks", components=("attn", "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out"),
                  file_layout=(("img_mlp.gate_layer.weight", "img_mlp.gate_up.weight", 0, 2),
                               ("img_mlp.proj.weight", "img_mlp.gate_up.weight", 1, 2)),
                  # measured 30 Sep on a 5090 at 0.5 MP: resident base 5.06 GB (MLP windows), streaming base 2.2;
                  # 0.5 -> 0.98 MP grew the peaks by up to +2.1 GB (~4 GB/MP)
                  overhead_gb=5.5, stream_base_gb=2.7, calib_mp=0.5, act_gb_per_mp=4.2)
```

## MiniMax H3: a clip-and-sound driver

### Caching in its own layout

`cache_stage` returning `True` takes the stage over: H3 caches video and audio latents and its text conditioning with its own scripts. Family options reach the cache only as `--aux` pairs.

<!-- excerpt: minimax/driver.py MiniMaxDriver.cache_stage -->
```python
def cache_stage(self, stage, datasets, args, device, aux):
    import argparse
    common = dict(skip_existing=args.skip_existing, keep_cache=args.keep_cache, num_workers=args.num_workers)
    if stage == "latents":
        from fizgig.scripts.minimax_cache_latents import cache_latents
        cache_latents(argparse.Namespace(vae=args.model, audio_vae=aux.get("audio_vae") or None,
                                         clip_still=aux.get("clip_still") == "1", batch_size=args.batch_size,
                                         **common), datasets, device)
    else:
        from fizgig.scripts.minimax_cache_text import cache_text
        # --aux reference_count=K directly, or the GUI's pair: distill_refs=K counted only with distill=1
        refs = int(aux.get("reference_count") or (aux.get("distill_refs", 0) if aux.get("distill") == "1" else 0))
        cache_text(argparse.Namespace(text_encoder=args.model, reference_count=refs,
                                      no_quantize=aux.get("no_quantize") == "1",
                                      batch_size=args.batch_size or 16, **common), datasets, device)
    return True
```

### Mapping its cache into the training conditioning

`batch_cond` turns one loaded item into what `training_loss` reads: caption dropout, reference-distillation teacher conditioning, the audio latent, and the voice-only flag.

<!-- excerpt: minimax/driver.py MiniMaxDriver.batch_cond -->
```python
def batch_cond(self, batch, device):
    import random
    text = batch["hidden_states"]
    if self._uncond is not None and random.random() < float(self.options.get("caption_dropout") or 0):
        text = self._uncond                                               # caption dropout step
    cond = {"hidden_states": text.to(device)}
    if "ref_hidden_states" in batch:          # reference distillation's teacher conditioning
        cond.update(ref_hidden_states=batch["ref_hidden_states"].to(device), ref_latent=batch["ref_latent"],
                    ref_token_tags=batch["ref_token_tags"][0])
    if batch.get("audio_latent") is not None:
        cond["audio_latent"] = batch["audio_latent"].to(device)
    if batch.get("audio_only") is not None and bool(batch["audio_only"].any()):
        cond["audio_only"] = True
    return cond
```

### A block map with named areas, that works with no model

The Profiler and Extract call `block_map()` with no model loaded, so it builds the names from the description; with a model it keeps only the modules that exist (the pruned base has fewer).

<!-- excerpt: minimax/driver.py MiniMaxDriver.block_map -->
```python
def block_map(self, dit=None):
    names = {n for n, _ in dit.named_modules()} if dit is not None else None

    def keep(mods):
        return [m for m in mods if names is None or m in names]
    main = [Block(f"h3blk_{i}", f"Block {i}", keep([f"blocks.{i}.{m}" for m in _BLOCK_MODULES]))
            for i in range(self.description.n_blocks)]
    refiner = [Block(f"h3_rf_{i}", f"Refiner {i}",
                     keep([f"token_refiner.blocks.{i}.{m}" for m in _BLOCK_MODULES])) for i in range(2)]
    return [BlockGroup("Blocks", main), BlockGroup("Token Refiner", refiner)]   # the old H3 panel's wording
```

### Making room for a decode on a small card

Previews decode on the training card. `park_for` moves only the missing gigabytes of tail blocks to CPU, because every extra gigabyte moved is a slower restore.

<!-- excerpt: minimax/driver.py MiniMaxDriver.park_for -->
```python
def park_for(self, dit, device, need_gb, purpose):
    """The old trainer's park: when the card cannot offer `need_gb` beside the resident base (the decode wants
    ~7.5 GB, an override encode the text encoder + 2), only the missing gigabytes (+1) of tail blocks go to CPU
    (park_dit_partial, which unbinds a streaming ring first) - every extra gigabyte moved is paging churn and a
    slower restore on a WDDM card."""
    from fizgig.minimax.common import park_dit_partial
    from fizgig.utils.device import plannable_free_vram
    need_gb = 7.5 if need_gb is None else need_gb
    gc.collect()
    torch.cuda.empty_cache()
    free = plannable_free_vram()
    if free >= need_gb:
        return False
    need = (need_gb - free) + 1.0
    logger.info(f"[preview] {free:.1f} GB free is too tight for {purpose} - parking ~{need:.1f} GB of tail blocks "
                f"for this pass.")
    park_dit_partial(dit, need_gb=need)
    gc.collect()
    torch.cuda.empty_cache()
    return True
```

### Writing a clip preview

A clip preview is frames, a wav and a playable mp4, then the middle frame as the PNG, written last: the gallery takes the PNG as its "finished" signal.

<!-- excerpt: minimax/driver.py MiniMaxDriver.save_preview -->
```python
def save_preview(self, result, path):
    """The old previews' contract: a clip writes every 2nd frame as JPEG in <stem>.clip/, the wav and a playable
    mp4 beside it, then the middle frame as the PNG - LAST, the gallery's 'finished' signal."""
    import os
    from PIL import Image
    if not isinstance(result, dict):
        result.save(path)
        return [path]
    stem = path[:-4]
    px, out = result["frames"], []
    clip_dir = stem + ".clip"
    os.makedirs(clip_dir, exist_ok=True)
    keep = list(range(0, px.shape[1], 1 if result.get("every_frame") else 2))
    if keep[-1] != px.shape[1] - 1:
        keep.append(px.shape[1] - 1)
    for k in keep:
        fr = (px[:, k].permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
        Image.fromarray(fr).save(os.path.join(clip_dir, f"f{k:03d}.jpg"), quality=87)
    if result.get("wave") is not None:
        from fizgig.minimax.common import write_preview_mp4, write_wav
        write_wav(stem + ".wav", result["wave"])
        out.append(stem + ".wav")
        try:
            write_preview_mp4(stem + ".mp4", px, stem + ".wav")
            out.append(stem + ".mp4")
        except Exception:
            pass                                     # the wav and the scrub frames still work
    elif result.get("every_frame"):
        # a clip slider's strip: the motion is the dial, so a playable mp4 at the true frame rate, silent
        from fizgig.minimax.common import write_preview_mp4
        try:
            write_preview_mp4(stem + ".mp4", px, None)
            out.append(stem + ".mp4")
        except Exception:
            pass
    result["image"].save(path)
    return out + [path]
```
