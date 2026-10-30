"""ACE-Step 1.5 XL driver for Fizgig's standard layer (families/driver.py) - text-to-music: songs with vocals and
lyrics, 48 kHz stereo. Two bases: XL Turbo (default) and XL SFT (family option base=sft).

* model: acestep/model.py (adapted from the official repository), loaded from ComfyUI's single file; the DiT is the
  root module, its condition encoder (text projector, lyric and timbre encoders) and null condition beside it
* conditioning, cached per item: Qwen3-Embedding-0.6B's last hidden state for the official text-to-music prompt
  (instruction, caption, bpm / time signature / key / duration) -> {"text": (Lt, 1024)}, and the token embeddings of
  "# Languages / # Lyric" -> {"lyrics": (Ll, 1024)}. The condition encoder runs inside every forward (frozen, no
  grad), with the silence latent as the timbre reference - the official handler's text-to-music without a reference
* latents: the Oobleck VAE (acestep/vae.py), 64 channels at 25 per second, (64, T); songs over MAX_SECONDS are cut
* training: rectified flow, x_t = t noise + (1 - t) x0, target noise - x0, unweighted MSE. Turbo trains on its own 8
  timesteps (shift 3), as the official trainer and Side-Step's turbo presets do; SFT on continuous logit-normal
  timesteps (mu -0.4, sigma 1) with 15% of steps conditioned on the null embedding (Side-Step's corrected mode)
* sampling: Euler; Turbo over the official schedule (8 steps, shift 3, no CFG), SFT 30 steps at shift 1 with APG
  guidance 7 against the null condition (the XL SFT code's defaults); context = the silence latent + ones
* LoRA: q/k/v/o of each layer's self- and cross-attention (the official trainer's targets), keys
  base_model.model.layers.N.self_attn.q_proj.lora_A.weight + .alpha, which ComfyUI loads as ACE-Step 1.5 LoRAs
"""
import json
import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from fizgig.families.driver import Block, BlockGroup, FamilyDriver

logger = logging.getLogger(__name__)

DTYPE = torch.bfloat16
HELPER = "ACE-Step/Ace-Step1.5"                    # tokenizer + silence latent (helper_files)
TOKENIZER = "Qwen3-Embedding-0.6B"
SILENCE = "acestep-v15-turbo/silence_latent.pt"
RATE = 25                                          # latent frames per second (48 kHz / hop 1920)
MAX_SECONDS = 240                                  # the official trainer's max_duration
REFER_FRAMES = 750                                 # the handler's timbre reference with no reference audio: 30 s
TEXT_TOKENS, LYRIC_TOKENS = 256, 2048              # the handler's truncation
INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"   # text2music (constants.py)
COVER_INSTRUCTION = "Generate audio semantic tokens based on the given conditions:"   # cover (constants.py)
REPAINT_INSTRUCTION = "Repaint the mask area based on the given conditions:"         # repaint (constants.py)
REFER_SEGMENT = 10                                 # the handler's reference: three 10 s pieces, front / middle / back
TURBO_T = (1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3)
SFT_STEPS, SFT_SHIFT, SFT_GUIDANCE = 30, 1.0, 7.0          # acestep-v15-xl-sft generate_audio defaults
TIMESTEP_MU, TIMESTEP_SIGMA, CFG_DROPOUT = -0.4, 1.0, 0.15  # config.json; Side-Step's corrected training
LYRICS_SUFFIX = ".lyrics.txt"
INSTRUMENTAL = "[Instrumental]"
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def caption_prompt(caption, seconds, bpm=None, timesignature=None, keyscale=None, instruction=INSTRUCTION):
    """The official SFT_GEN_PROMPT with the text-to-music instruction (or the cover one) and _dict_to_meta_string's
    metadata."""
    metas = (f"- bpm: {bpm or 'N/A'}\n- timesignature: {timesignature or 'N/A'}\n- keyscale: {keyscale or 'N/A'}\n"
             f"- duration: {int(seconds)} seconds\n")
    return f"# Instruction\n{instruction}\n\n# Caption\n{caption}\n\n# Metas\n{metas}<|endoftext|>\n"


def lyrics_prompt(lyrics, language="en"):
    """The handler's _format_lyrics."""
    return f"# Languages\n{language or 'en'}\n\n# Lyric\n{lyrics or INSTRUMENTAL}<|endoftext|>"


def song_info(audio_path):
    """A training song's sidecars: <stem>.lyrics.txt (else instrumental) and an optional <stem>.json with bpm,
    timesignature, keyscale, language."""
    stem = os.path.splitext(audio_path)[0]
    info = {}
    if os.path.exists(stem + ".json"):
        with open(stem + ".json", encoding="utf-8") as f:
            info = {k: v for k, v in json.load(f).items() if v not in (None, "")}
    if os.path.exists(stem + LYRICS_SUFFIX):
        with open(stem + LYRICS_SUFFIX, encoding="utf-8") as f:
            info["lyrics"] = f.read().strip()
    return info


def parse_prompt(prompt):
    """A sample prompt line: "<caption>" or "<caption> | <lyrics>", lyric lines separated by " / "."""
    caption, _, lyrics = prompt.partition("|")
    lyrics = "\n".join(s.strip() for s in lyrics.split(" / ")).strip()
    return caption.strip(), lyrics or INSTRUMENTAL


def read_audio(path, seconds=MAX_SECONDS):
    """Any audio file -> float32 (2, n) at 48 kHz, the first `seconds` (ffmpeg decodes and resamples)."""
    import subprocess
    from fizgig.families.clips import ffmpeg
    cmd = [ffmpeg(), "-v", "error", "-i", path, "-t", str(seconds), "-f", "f32le", "-ac", "2", "-ar", "48000", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()


def audio_seconds(path):
    """A song's length as trained (capped at MAX_SECONDS), from ffmpeg's banner - no decode."""
    import re
    import subprocess
    from fizgig.families.clips import ffmpeg
    out = subprocess.run([ffmpeg(), "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", out)
    if not m:
        raise ValueError(f"{os.path.basename(path)}: not a readable audio file")
    h, mi, se = m.groups()
    return min(MAX_SECONDS, int(h) * 3600 + int(mi) * 60 + float(se))


class _Momentum:
    """APG's momentum buffer (apg_guidance.MomentumBuffer, momentum -0.75)."""

    def __init__(self, momentum=-0.75):
        self.momentum, self.average = momentum, 0

    def update(self, value):
        self.average = value + self.momentum * self.average


def apg(cond, uncond, scale, momentum, norm_threshold=2.5):
    """Adaptive projected guidance (the XL SFT code's apg_forward, eta 0, dims [1] = time): the momentum-averaged
    (cond - uncond), clamped to norm 2.5 along time, minus its part parallel to cond."""
    momentum.update(cond - uncond)
    diff = momentum.average
    diff = diff * torch.minimum(torch.ones_like(diff), norm_threshold / diff.norm(p=2, dim=1, keepdim=True))
    c = F.normalize(cond.double(), dim=1)
    d = diff.double()
    orthogonal = (d - (d * c).sum(dim=1, keepdim=True) * c).to(cond.dtype)
    return cond + (scale - 1) * orthogonal


def _helper_file(name):
    from fizgig.utils import hf_public
    return hf_public.download(repo_id=HELPER, filename=name)


def silence_latent():
    """The official silence latent, (1, 15000, 64) float32."""
    return torch.load(_helper_file(SILENCE), map_location="cpu").transpose(1, 2).float()


class _TextEncoder:
    """Qwen3-Embedding-0.6B from ComfyUI's single file, with its tokenizer from the ACE-Step repository."""

    def __init__(self, path, device):
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Qwen3Config, Qwen3Model
        self.tok = AutoTokenizer.from_pretrained(HELPER, subfolder=TOKENIZER)
        model = Qwen3Model(Qwen3Config(
            vocab_size=151669, hidden_size=1024, intermediate_size=3072, num_hidden_layers=28,
            num_attention_heads=16, num_key_value_heads=8, head_dim=128, max_position_embeddings=32768,
            rms_norm_eps=1e-6, rope_theta=1000000.0, tie_word_embeddings=True, use_sliding_window=False))
        sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in load_file(path).items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"not ACE-Step's Qwen3 0.6B file: missing {missing[:5]}, unexpected {unexpected[:5]}")
        self.model = model.to(device, DTYPE).eval().requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def _text(self, caption_text):
        c = self.tok(caption_text, truncation=True, max_length=TEXT_TOKENS, return_tensors="pt").input_ids
        return self.model(input_ids=c.to(self.device)).last_hidden_state[0].to(DTYPE).cpu()

    @torch.no_grad()
    def encode(self, caption_text, lyric_text, cover_text=None, repaint_text=None):
        """-> {"text", "lyrics"} (+ "text_cover" / "text_repaint": the caption under the cover / repaint
        instruction, when given)."""
        ly = self.tok(lyric_text, truncation=True, max_length=LYRIC_TOKENS, return_tensors="pt").input_ids
        lyrics = self.model.embed_tokens(ly.to(self.device))[0]
        out = {"text": self._text(caption_text), "lyrics": lyrics.to(DTYPE).cpu()}
        if cover_text is not None:
            out["text_cover"] = self._text(cover_text)
        if repaint_text is not None:
            out["text_repaint"] = self._text(repaint_text)
        return out

    def unload(self):
        self.model.to("cpu")
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class AceStepDriver(FamilyDriver):
    _silence = None

    @property
    def silence(self):
        if AceStepDriver._silence is None:
            AceStepDriver._silence = silence_latent()
        return AceStepDriver._silence

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path, device):
        from accelerate import init_empty_weights
        from safetensors.torch import load_file
        from fizgig.acestep.model import AceConfig, AceStep15
        with init_empty_weights():
            dit = AceStep15(AceConfig())
        missing, unexpected = dit.load_checkpoint({k: v.to(DTYPE) for k, v in load_file(path).items()}, assign=True)
        if missing or unexpected:
            raise RuntimeError(f"not an ACE-Step 1.5 file: missing {missing[:5]}, unexpected {unexpected[:5]}")
        self._dit_file = path
        return dit.to(device).eval().requires_grad_(False)

    _guide = None                       # (model file, CoverGuide): read from the DiT's file on a cover's first use

    def cover_guide(self, device):
        from fizgig.acestep.model import CoverGuide
        path = getattr(self, "_dit_file", None)
        if path is None:
            raise RuntimeError("load the DiT first: the cover guide is read from its file")
        if AceStepDriver._guide is None or AceStepDriver._guide[0] != path:
            AceStepDriver._guide = None
            AceStepDriver._guide = (path, CoverGuide.from_checkpoint(path, device, DTYPE))
        g = AceStepDriver._guide[1]
        if next(g.parameters()).device != torch.device(device):
            g.to(device)
        return g

    def load_vae(self, path, device):
        from fizgig.acestep.vae import load_vae
        return load_vae(path, device)

    def load_text_encoder(self, path, device):
        return _TextEncoder(path, device)

    def unload_text_encoder(self, te):
        te.unload()

    def enable_gradient_checkpointing(self, dit, on=True):
        dit.gradient_checkpointing = bool(on)

    # ---- caching --------------------------------------------------------------------------------
    def cache_stage(self, stage, datasets, args, device, aux):
        from fizgig.families.cache import _clean, FORMAT_VERSION
        from safetensors import safe_open
        from safetensors.torch import save_file
        desc = self.description
        if stage == "latents":
            from fizgig.acestep import vae as V
            from fizgig.scripts.cache_latents import encode_datasets
            vae = self.load_vae(args.model, device)
            gen = torch.Generator().manual_seed(0)
            args.needs_reencode = lambda p: True       # skip_existing is decided below, by the source file

            def encode(batch):
                for it in batch:
                    stamp = f"{os.path.getsize(it.item_key)}:{int(os.path.getmtime(it.item_key))}:{MAX_SECONDS}"
                    if args.skip_existing and os.path.exists(it.latent_cache_path):
                        with safe_open(it.latent_cache_path, framework="pt") as f:
                            if (f.metadata() or {}).get("source") == stamp:
                                continue
                    wav = read_audio(it.item_key)
                    z = V.encode(vae, torch.from_numpy(wav), gen)[0].T            # (64, T)
                    logger.info(f"latent cache: {it.item_key} -> {tuple(z.shape)} ({wav.shape[1] / 48000:.1f} s)")
                    os.makedirs(os.path.dirname(it.latent_cache_path), exist_ok=True)
                    save_file({f"latent_{z.shape[1]}": _clean(z, "latent", it.item_key)}, it.latent_cache_path,
                              metadata={"architecture": desc.arch_id, "seconds": f"{wav.shape[1] / 48000:.3f}",
                                        "source": stamp, "format_version": FORMAT_VERSION})
            encode_datasets(datasets, encode, args)
            return True

        from fizgig.scripts.cache_text import post_process, prepare_cache_files_and_paths, process_batches
        files, paths = prepare_cache_files_and_paths(datasets)
        te = self.load_text_encoder(args.model, device)
        skip = args.skip_existing
        args.skip_existing = False                     # existence alone is not enough: the prompt is checked below

        def encode(batch):
            for it in batch:
                info = song_info(it.item_key)
                cap = caption_prompt(it.caption, int(audio_seconds(it.item_key)), info.get("bpm"),
                                     info.get("timesignature"), info.get("keyscale"))
                lyr = lyrics_prompt(info.get("lyrics"), info.get("language", "en"))
                key = cap + "\n" + lyr
                p = it.text_encoder_output_cache_path
                if skip and os.path.exists(p):
                    with safe_open(p, framework="pt") as f:
                        if (f.metadata() or {}).get("caption1") == key:
                            continue
                c = te.encode(cap, lyr)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                save_file({f"cond__{k}": _clean(v, k, it.item_key) for k, v in c.items()}, p,
                          metadata={"architecture": desc.arch_id, "caption1": key, "format_version": FORMAT_VERSION})
                logger.info(f"text cache: {it.item_key} ({c['text'].shape[0]} caption + {c['lyrics'].shape[0]} "
                            f"lyric tokens)")
        process_batches(args, datasets, files, paths, encode)
        self.unload_text_encoder(te)
        post_process(datasets, files, paths, args.keep_cache)
        return True

    def media_problem(self, path):
        stem = os.path.splitext(path)[0]
        if os.path.exists(stem + ".json"):
            try:
                with open(stem + ".json", encoding="utf-8") as f:
                    json.load(f)
            except ValueError as e:
                return f"{os.path.basename(stem)}.json is not valid JSON ({e})"
        return ""

    def encode_text(self, te, captions):
        """Sample prompts ("<caption> | <lyrics>"), at the preview length."""
        seconds = self._preview_seconds()
        return [te.encode(caption_prompt(c, seconds), lyrics_prompt(ly),
                          caption_prompt(c, seconds, instruction=COVER_INSTRUCTION),
                          caption_prompt(c, seconds, instruction=REPAINT_INSTRUCTION))
                for c, ly in map(parse_prompt, captions)]

    def text_cache_key(self):
        return (self._preview_seconds(),)            # the caption says the length: 30 s and 60 s encode apart

    def _preview_seconds(self):
        return int(float(self.options.get("preview_seconds", 30)))

    # ---- the model call -------------------------------------------------------------------------
    def _condition(self, dit, cond, frames, device, refer=None, text_key="text"):
        """The condition encoder's sequence for one item (no grad: the encoder is never trained). refer: a reference
        song's latents (1, 750, 64) for the timbre encoder, the silence latent without one; text_key "text_cover":
        the caption under the cover instruction (a cover render's guided steps)."""
        text = cond.get(text_key, cond["text"]).to(device, DTYPE)
        lyrics = cond["lyrics"].to(device, DTYPE)
        text = text if text.dim() == 3 else text[None]
        lyrics = lyrics if lyrics.dim() == 3 else lyrics[None]
        refer = (self.silence[:, :REFER_FRAMES] if refer is None else refer).to(device, DTYPE)
        with torch.no_grad():
            enc, _ = dit.condition(text, torch.ones(text.shape[:2], device=device), lyrics,
                                   torch.ones(lyrics.shape[:2], device=device), refer)
        return enc

    def _context(self, frames, device):
        from fizgig.acestep.model import context_latents
        return context_latents(self.silence, frames).to(device, DTYPE)

    # ---- training -------------------------------------------------------------------------------
    @staticmethod
    def _x0(latents):
        """Cached (1, 64, T) -> the model's (1, T, 64)."""
        return latents.float().transpose(1, 2)

    @property
    def sft(self):
        """Training (or previewing) on the XL SFT base (family option base=sft)."""
        return self.options.get("base") == "sft"

    def _draw_t(self, n, generator, min_t, max_t):
        """Turbo: its own 8 timesteps, uniformly (the official trainer); a window keeps those inside. SFT: continuous
        logit-normal (mu -0.4, sigma 1), rescaled into the window."""
        if self.sft:
            t = torch.sigmoid(torch.randn(n, generator=generator) * TIMESTEP_SIGMA + TIMESTEP_MU)
            return min_t + (max_t - min_t) * t
        ts = [t for t in TURBO_T if min_t <= t <= max_t] or list(TURBO_T)
        idx = torch.randint(len(ts), (n,), generator=generator)
        return torch.tensor([ts[i] for i in idx])

    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                      diff_weight=0.0):
        x0 = self._x0(latents)
        dev = x0.device
        t = self._draw_t(x0.shape[0], generator, min_t, max_t).to(dev)
        noise = torch.randn(x0.shape, generator=generator).to(dev)
        tb = t.view(-1, 1, 1)
        xt = tb * noise + (1 - tb) * x0
        enc = self._condition(dit, cond, x0.shape[1], dev)
        if self.sft and float(torch.rand(1, generator=generator)) < CFG_DROPOUT:
            enc = dit.null_condition_emb.to(enc).expand_as(enc)      # the guidance's null condition, learned too
        pred = dit(xt.to(DTYPE), t.to(DTYPE), enc, self._context(x0.shape[1], dev)).float()
        return F.mse_loss(pred, noise - x0), {"t": float(t.mean())}

    def noise_latents(self, latents, generator, *, min_t=0.0, max_t=1.0):
        x0 = self._x0(latents)
        t = self._draw_t(1, generator, min_t, max_t).to(x0.device)
        noise = torch.randn(x0.shape, generator=generator).to(x0.device)
        return {"xt": t.view(-1, 1, 1) * noise + (1 - t.view(-1, 1, 1)) * x0, "t": t}

    def predict(self, dit, state, cond):
        xt = state["xt"]
        enc = self._condition(dit, cond, xt.shape[1], xt.device)
        return dit(xt.to(DTYPE), state["t"].to(DTYPE), enc, self._context(xt.shape[1], xt.device))

    # ---- sampling -------------------------------------------------------------------------------
    def _frames(self, seconds=None):
        return int(round((seconds or self._preview_seconds()) * RATE))

    @torch.no_grad()
    def initial_noise(self, seed, width=0, height=0, frames=None):
        g = torch.Generator("cpu").manual_seed(int(seed))
        return torch.randn((1, frames or self._frames(), 64), generator=g, dtype=torch.float32)

    supports_prompt_blend = True        # generate(cond_b=, prompt_blend=): Repair Studio's Prompt B
    supports_audio_source = True        # generate(source=, denoise=, follow=, refer=): Repair Studio's audio to
    #                                     audio (a song to start from / follow) and its reference (sound like)

    @torch.no_grad()
    def encode_reference(self, vae, path):
        """A reference recording -> the timbre encoder's input (1, 750, 64): the handler's 30 s of it, a 10 s piece
        from the front, middle and back thirds (shorter recordings repeat to 30 s). The official handler picks each
        piece's start at random; here each piece is the middle of its third, so a reference always sounds the same."""
        wav = read_audio(path, MAX_SECONDS)
        seg = REFER_SEGMENT * 48000
        if wav.shape[1] == 0 or not np.abs(wav).max() > 0:
            raise ValueError("the reference is silent")
        if wav.shape[1] < 3 * seg:
            wav = np.tile(wav, (1, math.ceil(3 * seg / wav.shape[1])))
        third = wav.shape[1] // 3
        parts = []
        for k in range(3):
            lo, hi = k * third, (wav.shape[1] if k == 2 else (k + 1) * third)
            start = lo + max(0, (hi - lo - seg) // 2)
            parts.append(wav[:, start:start + seg])
        x = torch.from_numpy(np.concatenate(parts, 1).copy()).to(vae.device, vae.dtype)[None]
        lat = vae.encode(x).latent_dist.mean.transpose(1, 2).float()
        return lat[:, :REFER_FRAMES].cpu()

    @staticmethod
    def source_wave(path, seconds):
        """The source song as a render uses it: the first `seconds` at 48 kHz, (2, n), zero-padded when shorter."""
        wav = read_audio(path, seconds)
        n = int(round(float(seconds) * 48000))
        if wav.shape[1] < n:
            wav = np.pad(wav, ((0, 0), (0, n - wav.shape[1])))
        return np.ascontiguousarray(wav[:, :n])

    @staticmethod
    def repaint_frames(region, frames):
        """(start, end) seconds -> the official latent span [a, b): int(sec * 48000 // 1920), clamped, b > a."""
        a = max(0, min(int(float(region[0]) * 48000 // 1920), frames - 1))
        return a, max(a + 1, min(int(float(region[1]) * 48000 // 1920), frames))

    @staticmethod
    def repaint_config(strength):
        """The handler's balanced mode (_resolve_repaint_config): strength s -> (the share of the steps that hold
        the rest of the song to the source, latent crossfade frames, waveform crossfade seconds)."""
        inv = 1.0 - min(max(float(strength), 0.0), 1.0)
        return inv, round(25 * inv), 0.05 * inv

    @staticmethod
    def splice_source(wave, source_wave, region, strength):
        """apply_repaint_waveform_splice: the source's own waveform outside the repainted section (no VAE round
        trip there), a linear crossfade at its edges. wave / source_wave (2, n) torch at 48 kHz."""
        n = min(wave.shape[-1], source_wave.shape[-1])
        a = max(0, min(int(float(region[0]) * 48000), n))
        b = max(a, min(int(float(region[1]) * 48000), n))
        if a == 0 and b >= n:
            return wave
        cf = int(AceStepDriver.repaint_config(strength)[2] * 48000)
        m = torch.zeros(n)
        m[a:b] = 1.0
        if cf > 0:
            lo = max(a - cf, 0)
            if a - lo > 0:
                m[lo:a] = torch.linspace(0.0, 1.0, a - lo + 2)[1:-1]
            hi = min(b + cf, n)
            if hi - b > 0:
                m[b:hi] = torch.linspace(1.0, 0.0, hi - b + 2)[1:-1]
        src = source_wave[:, :n].to(wave.dtype)
        out = m * wave[:, :n] + (1.0 - m) * src
        return torch.cat([out, wave[:, n:]], dim=-1) if wave.shape[-1] > n else out

    @torch.no_grad()
    def encode_source(self, vae, path, seconds):
        """Any audio (or video) file -> its latent (1, T, 64) at the render's length: the first `seconds`, padded
        with silence when shorter. The posterior's mean, so the same file always gives the same start."""
        x = torch.from_numpy(self.source_wave(path, seconds)).to(vae.device, vae.dtype)[None]
        lat = vae.encode(x).latent_dist.mean.transpose(1, 2).float()
        frames = self._frames()
        if lat.shape[1] < frames:
            lat = torch.cat([lat, self.silence[:, :frames - lat.shape[1]].to(lat)], 1)
        return lat[:, :frames].cpu()

    @staticmethod
    def source_start(ts, denoise):
        """The first of the schedule's timesteps at or below `denoise` (where an audio-to-audio render starts), or
        None when even the last is above it (the source comes back unchanged)."""
        return next((t for t in ts if t <= float(denoise) + 1e-6), None)

    def schedule(self, steps, sigmas=None, options=()):
        """The timesteps a render runs (before the final 0)."""
        if self.sft:
            ts = [1.0 - i / SFT_STEPS for i in range(SFT_STEPS)]
            return [SFT_SHIFT * t / (1 + (SFT_SHIFT - 1) * t) for t in ts] if SFT_SHIFT != 1.0 else ts
        if sigmas is not None:
            return [float(s) for s in sigmas if float(s) > 0.0]
        if int(steps) == len(TURBO_T):
            return list(TURBO_T)
        shift = float(dict(options or ()).get("shift", 3.0))
        return [shift * t / (1 + (shift - 1) * t) for t in (1.0 - i / int(steps) for i in range(int(steps)))]

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None, cond_b=None, prompt_blend=0.0, source=None, denoise=1.0,
                 follow=0.0, refer=None, repaint=None, repaint_strength=0.5, **_ignored):
        """cond_b + prompt_blend w: every step runs the model with both prompts and mixes the predictions,
        (1 - w) * A + w * B (guided SFT: each guided on its own, then mixed) - exact at w 0 (A) and 1 (B), any prompt
        lengths. source + denoise d (audio to audio): the render starts from the source latent noised to the first
        timestep at or below d, and runs only the steps from there (d 1 = from pure noise, as without a source).
        follow f (the official cover, audio_cover_strength): the first int(steps * f) of the steps run follow the
        source's melody guide (its 5 Hz codes, CoverGuide) under the cover instruction, the rest are text to music.
        refer: a reference song's latents for the timbre encoder (sound like), every step.
        repaint (start, end) seconds (the official repaint; past the source's end = extend): only that section is
        made new, from noise, under the repaint instruction; the rest of the song is the source's, held to it
        (noised to each step's level) for the first share of the steps, its edges crossfaded in the latent
        (repaint_strength: the handler's balanced mode, 0.5 its default). denoise and follow do not apply."""
        device = next(dit.parameters()).device
        w = min(max(float(prompt_blend or 0.0), 0.0), 1.0) if cond_b is not None else 0.0
        if w >= 1.0:
            cond, w = cond_b, 0.0                   # all B: one pass
        x = (self.initial_noise(seed) if noise is None else noise).to(device).float()
        guided = self.sft
        ts = self.schedule(steps, sigmas, options)
        rp = None
        if source is not None and repaint is not None:
            clean = source.to(device).float()
            a, b = self.repaint_frames(repaint, clean.shape[1])
            mask = torch.zeros(clean.shape[1], dtype=torch.bool, device=device)
            mask[a:b] = True
            src = clean.clone()
            src[:, a:b] = self._context(clean.shape[1], device)[:, a:b, :64].float()   # the tiled silence latent
            inj, cf, _ = self.repaint_config(repaint_strength)
            rp = (clean, mask, torch.cat([src, mask[None, :, None].float().expand_as(src)], -1).to(DTYPE),
                  round(inj * len(ts)), cf, mask[None, :, None].expand_as(clean), x.clone())
            denoise, follow = 1.0, 0.0
        if source is not None and float(denoise) < 1.0:
            src = source.to(device).float()
            t0 = self.source_start(ts, denoise)
            if t0 is None:
                return src                          # below the last step: the source as it is
            ts = [t for t in ts if t <= t0 + 1e-9]
            x = t0 * x + (1.0 - t0) * src           # the source, noised to where the render starts
        ts = ts + [0.0]
        f = min(max(float(follow or 0.0), 0.0), 1.0) if source is not None else 0.0
        cover_steps = int((len(ts) - 1) * f)

        def stage(cover):
            if rp is not None:
                ctx = rp[2]
            elif cover:
                hint = self.cover_guide(device)(source.to(device, DTYPE), self.silence.to(device, DTYPE))
                ctx = torch.cat([hint, torch.ones_like(hint)], dim=-1)
            else:
                ctx = self._context(x.shape[1], device)
            key = "text_repaint" if rp is not None else "text_cover" if cover else "text"
            ps = [self._condition(dit, cond, x.shape[1], device, refer, key)]
            if w > 0.0:
                ps.append(self._condition(dit, cond_b, x.shape[1], device, refer, key))
            if guided:
                ps = [torch.cat([e, dit.null_condition_emb.to(e).expand_as(e)]) for e in ps]
                ctx = ctx.expand(2, -1, -1)
            return ctx, ps

        ctx, prompts = stage(cover_steps > 0)
        if guided:
            momenta = [_Momentum() for _ in prompts]
        for i in range(len(ts) - 1):
            if on_step is not None:
                on_step(i, len(ts) - 1)
            if i == cover_steps and i > 0:
                ctx, prompts = stage(False)         # the rest of the steps: text to music
            xin = torch.cat([x, x]) if guided else x
            t = torch.full((xin.shape[0],), ts[i], device=device, dtype=DTYPE)
            vs = []
            for k, enc in enumerate(prompts):
                v = dit(xin.to(DTYPE), t, enc, ctx).float()
                vs.append(apg(v[:1], v[1:], SFT_GUIDANCE, momenta[k]) if guided else v)
            v = vs[0] if len(vs) == 1 else (1.0 - w) * vs[0] + w * vs[1]
            x = x + (ts[i + 1] - ts[i]) * v
            if rp is not None and i < rp[3] and i < len(ts) - 2:
                # _repaint_step_injection: the rest of the song back to the source at the next step's noise level
                tn = ts[i + 1]
                x = torch.where(rp[5], x, tn * rp[6] + (1.0 - tn) * rp[0])
        if rp is not None and rp[4] > 0:
            x = self._repaint_blend(x, rp[0], rp[1], rp[4])
        return x

    @staticmethod
    def _repaint_blend(x, clean, mask, cf):
        """_repaint_boundary_blend: the soft mask (1 in the section, linear ramps cf frames into the rest)."""
        soft = mask.float().clone()
        idx = torch.nonzero(mask).squeeze(-1)
        if 0 < idx.numel() < mask.numel():
            left, right, T = int(idx[0]), int(idx[-1]) + 1, mask.numel()
            lo = max(left - cf, 0)
            if left - lo > 0:
                soft[lo:left] = torch.linspace(0.0, 1.0, left - lo + 2, device=soft.device)[1:-1]
            hi = min(right + cf, T)
            if hi - right > 0:
                soft[right:hi] = torch.linspace(1.0, 0.0, hi - right + 2, device=soft.device)[1:-1]
        s = soft[None, :, None].expand_as(x)
        return s * x + (1.0 - s) * clean

    @torch.no_grad()
    def decode(self, vae, latents, width=0, height=0):
        """-> a Song (waveform + a picture of it), which save_preview writes."""
        from fizgig.acestep import vae as V
        lat = latents if latents.shape[-1] == 64 else latents.transpose(1, 2)
        return Song(V.decode(vae, lat.float()))

    def decode_audio(self, vae, audio):
        return self.decode(vae, audio).wave

    def practice_latent(self, vae, latents, res):
        """A prompt slider's practice song stays a latent: (1, T, 64) from generate() -> the cache's (1, 64, T)."""
        return latents.float().transpose(1, 2).contiguous().cpu()

    def slider_preview(self, frames, multipliers):
        """The dial as one song: -1, 0 and +1 one after another, a second of silence between them."""
        gap = torch.zeros(frames[0].wave.shape[0], 48000, dtype=frames[0].wave.dtype)
        parts = []
        for k, f in enumerate(frames):
            parts += ([gap] if k else []) + [f.wave]
        return Song(torch.cat(parts, dim=-1))

    def save_preview(self, result, path):
        from fizgig.minimax.common import write_wav
        from fizgig.acestep.vae import SAMPLE_RATE
        wav = os.path.splitext(path)[0] + ".wav"
        write_wav(wav, result.wave, sample_rate=SAMPLE_RATE)
        result.picture().save(path)                  # last: the gallery's "finished" signal
        return [wav, path]

    # ---- LoRA and the block map -------------------------------------------------------------------
    def block_map(self, dit=None):
        names = {n for n, _ in dit.named_modules()} if dit is not None else None
        blocks = []
        for i in range(self.description.n_blocks):
            mods = [f"layers.{i}.{a}.{p}" for a in ("self_attn", "cross_attn") for p in ATTN]
            if names is not None:
                mods = [m for m in mods if m in names]
            blocks.append(Block(f"layer_{i}", f"Layer {i}", mods))
        return [BlockGroup("DiT layers", blocks)]

    def activity_axis(self, module_name):
        """The song's time axis: every self-attention projection, and cross-attention's query and output (its key
        and value project the caption and lyrics)."""
        return ".self_attn." in module_name or module_name.endswith(("cross_attn.q_proj", "cross_attn.o_proj"))

    def quant_target_names(self, dit):
        """The 4-bit trunk a fine-tune trains against: both attentions and the MLP of every layer (the base is bf16
        otherwise - nothing else quantises)."""
        return [f"layers.{i}.{a}.{p}" for i in range(self.description.n_blocks)
                for a, ps in (("self_attn", ATTN), ("cross_attn", ATTN), ("mlp", MLP)) for p in ps]

    def ft_spec(self, dit):
        """Fine-tuning: the 32 layers' self-attention, cross-attention and MLP Linears on the NF4 trunk, in five
        windows of similar size per layer (each attention ~26M parameters, each MLP matrix ~25M). The condition
        encoder, the AdaLN tables, norms and I/O stay frozen. The file names the layers "decoder.layers.N"; the
        checkpoint is the source file with those tensors replaced, so ComfyUI loads it as it loads the base."""
        from fizgig.families.ft import FTSpec
        return FTSpec(blocks="layers", components=("self_attn", "cross_attn", "mlp.gate_proj", "mlp.up_proj",
                                                   "mlp.down_proj"),
                      file_prefix="decoder.")

    def alias_flat(self, flat):
        """ComfyUI's checkpoint naming (decoder_layers_N_...) -> the decoder-rooted name."""
        return flat[len("decoder_"):] if flat.startswith("decoder_layers_") else None


class Song:
    """A decoded preview: the waveform (2, n) at 48 kHz, and a waveform picture for the gallery tile."""

    def __init__(self, wave):
        self.wave = wave

    def picture(self, width=1024, height=256):
        from PIL import Image, ImageDraw
        w = self.wave.mean(0).numpy()
        cols = np.array_split(np.abs(w), width)
        peak = max(1e-6, float(np.abs(w).max()))
        img = Image.new("RGB", (width, height), (24, 26, 33))
        d = ImageDraw.Draw(img)
        mid = height // 2
        for x, c in enumerate(cols):
            a = float(c.max()) / peak if c.size else 0.0
            r = float(np.sqrt((c ** 2).mean())) / peak if c.size else 0.0
            d.line([(x, mid - a * mid * 0.95), (x, mid + a * mid * 0.95)], fill=(70, 110, 170))
            d.line([(x, mid - r * mid * 0.95), (x, mid + r * mid * 0.95)], fill=(140, 190, 250))
        return img
