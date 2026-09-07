"""MiniMax H3 caching: video-VAE latents (24-ch) + Qwen3-VL-32B text-encoder states.

Reuses Fizgig's dataset framework (ItemInfo + safetensors cache files, same as Krea 2); only
the encode + the cache contents are H3-specific:

* Latents: the H3 video VAE encodes a still image as one frame -> [1, 24, 1, H/16, W/16]. The
  1-frame axis is squeezed for storage -> (24, H/16, W/16), keyed `latent_{H}x{W}` so the shared
  collate maps it to the batch's `latents`. The trainer re-adds the frame axis before the DiT.
* Text: the H3 conditioning is the raw layer-50 Qwen3-VL output -> (L, 5120). Stored as
  `hidden_states` (L, 5120) plus an all-valid `attention_mask` (L,), matching the Krea 2 text
  cache keys so the same collate path applies (batch size 1 -> no padding needed).
"""

import logging
import os
from typing import List

import numpy as np
import torch
from safetensors.torch import save_file

from fizgig.dataset.image_dataset import ItemInfo, MemoryEfficientSafeOpen, dtype_to_str
from fizgig.training.metadata import ARCHITECTURE_MINIMAX

logger = logging.getLogger(__name__)


# --- cache writers -----------------------------------------------------------
def save_latent_cache_minimax(item_info: ItemInfo, latent: torch.Tensor,
                              audio_latent: torch.Tensor = None,
                              audio_only: bool = False,
                              still_latent: torch.Tensor = None,
                              still_frame: int = None) -> None:
    """Save an H3 VAE latent to the item's cache file, optionally with the clip's audio.

    Two shapes, two keys, and the still one is unchanged on purpose:

      * a still  -> 3-D (C, H, W)     keyed `latent_{H}x{W}`
      * a clip   -> 4-D (C, T, H, W)  keyed `latent_{T}x{H}x{W}`

    A still written today is byte-identical to one written before clips existed, so every cache
    already on disk stays valid and nobody re-caches a working dataset. Both keys start `latent_`,
    which is all the shared collate looks at (image_dataset.py: `startswith("latent_")` ->
    `latents`), and the trainer only adds a frame axis when one is missing — so a 4-D cache
    arrives at the DiT as (1, 24, T, H, W) with no further work.

    audio_latent is stored ALREADY PACKED into DiT rows — (2*T_audio, 32), channel-major — rather
    than in the VAE's own (32, 2, T) layout. Packing at write time means the ordering is decided
    once, next to the encoder that produced it, instead of being re-derived on every load where
    getting it backwards would train against scrambled targets and raise nothing.

    Absent audio is absent, not zeros: an item with no `audio_latent` key contributes no audio
    loss, which is what a muted clip and every still are meant to do. A silence TARGET would
    teach the model that this footage sounds like nothing.

    audio_only marks a VOICE item — its video latent is a zeros placeholder and the trainer
    zeroes the video loss for it. Stored as a cache key (not metadata) because keys are what
    the collate forwards into the batch; the flag rides beside the tensors it governs.
    """
    assert latent.dim() in (3, 4), \
        f"latent must be 3-D (C,H,W) or 4-D (C,T,H,W), got {tuple(latent.shape)}"
    if latent.dim() == 3:
        _, H, W = latent.shape
        key = f"latent_{H}x{W}"
    else:
        _, T, H, W = latent.shape
        key = f"latent_{T}x{H}x{W}"
    sd = {key: latent.detach().cpu().contiguous()}
    if audio_latent is not None:
        assert audio_latent.dim() == 2 and audio_latent.shape[1] == 32, (
            f"audio_latent must be packed rows (2*T, 32), got {tuple(audio_latent.shape)} — "
            f"use audio_vae.pack_audio() on the encoder's (B, 32, 2, T) output")
        sd["audio_latent"] = audio_latent.detach().cpu().contiguous()
    if audio_only:
        assert audio_latent is not None, "an audio-only item with no audio rows trains nothing"
        sd["audio_only"] = torch.tensor(True)
    if still_latent is not None:
        # The clip's chosen frame encoded as an ordinary still (C, H, W) — the "clip still as a
        # photo" training item. Keyed OFF `latent_` on purpose: the collate must never see it
        # as a second latent; __getitem__ swaps it in for the derived still item and drops it
        # for the clip itself. `still_frame` records which pixel frame it was.
        assert still_latent.dim() == 3, f"still_latent must be (C, H, W), got {tuple(still_latent.shape)}"
        sd["still_latent"] = still_latent.detach().cpu().contiguous()
        sd["still_frame"] = torch.tensor(int(still_frame if still_frame is not None else 0))
    for key, value in sd.items():
        if value.is_floating_point() and torch.isnan(value).any():
            logger.warning(f"NaN in {key} for {item_info.item_key} - replaced with 0")
            value[torch.isnan(value)] = 0
    metadata = {
        "architecture": ARCHITECTURE_MINIMAX,
        "width": str(item_info.original_size[0]),
        "height": str(item_info.original_size[1]),
        "dtype": dtype_to_str(latent.dtype),
        "format_version": "2.0.0",
    }
    os.makedirs(os.path.dirname(item_info.latent_cache_path), exist_ok=True)
    save_file(sd, item_info.latent_cache_path, metadata=metadata)


def save_text_encoder_output_cache_minimax(item_info: ItemInfo, hidden_states: torch.Tensor) -> None:
    """Save the raw Qwen3-VL-32B layer-50 states (L, 5120) + an all-valid mask (L,)."""
    hidden_states = hidden_states.detach().cpu().contiguous()
    if torch.isnan(hidden_states).any():
        logger.warning(f"NaN in hidden_states for {item_info.item_key} - replaced with 0")
        hidden_states[torch.isnan(hidden_states)] = 0
    sd = {
        "hidden_states": hidden_states,
        "attention_mask": torch.ones(hidden_states.shape[0], dtype=torch.bool),
    }
    metadata = {
        "architecture": ARCHITECTURE_MINIMAX,
        "caption1": item_info.caption,
        "dtype": dtype_to_str(hidden_states.dtype),
        "format_version": "2.0.0",
    }
    os.makedirs(os.path.dirname(item_info.text_encoder_output_cache_path), exist_ok=True)
    save_file(sd, item_info.text_encoder_output_cache_path, metadata=metadata)


def reference_cache_path(te_cache_path: str, slot: int = 0) -> str:
    """One r2v conditioning slot beside an item's ordinary text cache.

    `..._minimaxh3_te.safetensors` -> `..._minimaxh3_teref0.safetensors` (slot 0, 1, ...).

    SIBLING files, never a replacement: a distillation run needs both, since the student is
    conditioned on the plain caption and only the teacher gets the reference. Several slots
    because each item is paired with several OTHER dataset images — one reference per slot,
    picked at random per step, so the teacher's sequence stays short while the LoRA still sees
    every pairing across an epoch.

    The dataset's stale-cache glob is `*_{arch}_te.safetensors`, which matches none of these."""
    stem, ext = os.path.splitext(te_cache_path)
    return f"{stem}ref{int(slot)}{ext}"


def reference_cache_slots(te_cache_path: str, max_slots: int = 8):
    """Which reference slots actually exist for an item, as a list of ints."""
    return [i for i in range(max_slots) if os.path.exists(reference_cache_path(te_cache_path, i))]


def save_reference_text_cache(te_cache_path: str, slot: int, hidden_states: torch.Tensor,
                              token_tags: torch.Tensor, ref_latent: torch.Tensor,
                              caption: str = "", reference: str = "") -> str:
    """Save one slot: r2v conditioning (L, 5120), its per-row tags (L,), and the reference's
    own latent (24, h, w).

    All three travel together because they only mean anything together. The tags say which rows
    are the `<Picture 1>` vision block — states without them would have those rows silently
    modulated as text. The latent must be the SAME picture the vision block describes, or the
    teacher is shown one image and conditioned on another."""
    hidden_states = hidden_states.detach().cpu().contiguous()
    if torch.isnan(hidden_states).any():
        logger.warning(f"NaN in reference hidden_states for {te_cache_path} - replaced with 0")
        hidden_states[torch.isnan(hidden_states)] = 0
    path = reference_cache_path(te_cache_path, slot)
    sd = {"hidden_states": hidden_states,
          "token_tags": token_tags.detach().cpu().to(torch.int64).contiguous(),
          "ref_latent": ref_latent.detach().cpu().contiguous()}
    metadata = {"architecture": ARCHITECTURE_MINIMAX, "caption1": caption,
                "reference": os.path.basename(reference), "slot": str(slot),
                "dtype": dtype_to_str(hidden_states.dtype), "format_version": "2.0.0"}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(sd, path, metadata=metadata)
    return path


def load_reference_text_cache(te_cache_path: str, slot: int = 0):
    """-> (hidden_states, token_tags, ref_latent) for a slot, or None when it does not exist."""
    path = reference_cache_path(te_cache_path, slot)
    if not os.path.exists(path):
        return None
    # MemoryEfficientSafeOpen, not safetensors.safe_open — the repo uses it everywhere for the
    # documented Windows mmap crash, and these are read on the training hot path.
    with MemoryEfficientSafeOpen(path) as f:
        return (f.get_tensor("hidden_states"), f.get_tensor("token_tags"),
                f.get_tensor("ref_latent"))


def load_text_encoder_output_cache_minimax(path: str):
    """Return (hidden_states (L, 5120), attention_mask (L,) bool) from a cache file."""
    with MemoryEfficientSafeOpen(path) as f:
        return f.get_tensor("hidden_states"), f.get_tensor("attention_mask").to(torch.bool)


# --- clips -------------------------------------------------------------------
def _is_clip(item: ItemInfo) -> bool:
    """A clip is an item whose source file is a video AND whose content is many frames.

    Both halves matter: `content` being a list already means "multi-target" elsewhere in the
    dataset, so the extension is what says this list is time rather than alternatives.
    """
    from fizgig.minimax.clip import is_video
    return is_video(str(item.item_key)) and isinstance(item.content, list) and len(item.content) > 1


def _encode_clip(vae, item: ItemInfo, audio_vae, device, dtype, clip_still: bool = False) -> None:
    """One clip -> one cache file holding its video latent and, if it has sound, its audio rows.

    clip_still: also pick the clip's sharpest frame that shows a face (still_pick) and encode it
    as an ordinary still into the same file, for the "clip still as a photo" training item."""
    from fizgig.minimax.audio import HOP_LENGTH
    from fizgig.minimax.clip import read_audio
    from fizgig.minimax.model import audio_latents_for_frames

    frames = torch.from_numpy(np.stack(item.content))            # (T, H, W, C) uint8
    n_frames = frames.shape[0]
    x = frames.permute(3, 0, 1, 2).unsqueeze(0).float() / 127.5 - 1.0   # (1, C, T, H, W)
    with torch.no_grad():
        # encode_clip, not encode: the clip goes in as 17-frame groups, which is both what makes
        # the latent frame count match the DiT's clock and what keeps peak VRAM off the clip's
        # length. See MiniMaxH3VideoVAEEncoder.encode_clip.
        latent = vae.encode_clip(x.to(device, dtype=dtype))[0]   # (24, T', H/16, W/16)

    still_latent, still_frame = None, None
    if clip_still:
        from fizgig.minimax.still_pick import pick_still_frame
        still_frame, info = pick_still_frame(frames.numpy())
        with torch.no_grad():
            still_latent = vae.encode(x[:, :, still_frame:still_frame + 1].to(device, dtype=dtype))[0]
        still_latent = still_latent.squeeze(1) if still_latent.dim() == 4 else still_latent
        logger.info("[still] %s: frame %d of %d (%s, face sharpness %.0f, %d detections)",
                    os.path.basename(str(item.item_key)), still_frame, n_frames,
                    "sharpest with a face" if info["face"] else "NO FACE FOUND — frame 0",
                    info["score"], info["detections"])

    audio_rows = None
    if audio_vae is not None:
        wav = read_audio(str(item.item_key))                     # None when muted or silent
        if wav is not None:
            from fizgig.minimax.audio_vae import pack_audio

            # Fit the WAVEFORM to the frame count before encoding, rather than repairing rows
            # afterwards. A real container's audio rarely lines up with its video to the sample:
            # AAC granularity leaves a 22-frame clip about 20 ms short, which is one latent
            # short of the 37 the DiT will pack — and a row count the model does not expect is
            # a shifted target, not an error anyone would see.
            #
            # HOP-EXACT, never seconds-based: the target is the latent count the DiT demands
            # times the VAE's 800-sample hop — the voice path's trim_to_grid doctrine. The old
            # target, ceil(frames/24 x 32000) samples, disagrees with round(frames/24 x 40)
            # latents whenever the latent count rounds DOWN: a 56-frame clip got 74,667
            # samples, which encodes to 94 latents against the 93 required, and the belt-and-
            # braces check below refused every 5-, 56- and 107-frame clip with sound (Peter's
            # run died on the first 56). 22/39/73/90/124 agreed by rounding luck, which is why
            # it survived earlier clip training.
            want_samples = audio_latents_for_frames(n_frames) * HOP_LENGTH
            have = wav.shape[1]
            if have < want_samples:
                wav = np.pad(wav, ((0, 0), (0, want_samples - have)))
            elif have > want_samples:
                wav = wav[:, :want_samples]

            a_dev = next(audio_vae.parameters()).device
            a_dt = next(audio_vae.parameters()).dtype
            with torch.no_grad():
                z = audio_vae.encode(torch.from_numpy(wav).unsqueeze(0).to(a_dev, dtype=a_dt))
            audio_rows = pack_audio(z).float().cpu()

            # Belt and braces: the DiT sizes its audio block from the PIXEL frame count, so this
            # has to agree with the helper the model itself uses. Refuse rather than cache a
            # target the trainer will silently mis-read.
            want = audio_latents_for_frames(n_frames)
            got = audio_rows.shape[0] // 2
            if got != want:
                raise ValueError(
                    f"{item.item_key}: audio encoded to {got} latents but {n_frames} pixel "
                    f"frames need {want}. The clip's audio and video lengths disagree by more "
                    f"than padding can bridge — re-export it with Gizmo.")

    logger.info("latent cache: %s -> %s%s", item.item_key, tuple(latent.shape),
                f" + audio {tuple(audio_rows.shape)}" if audio_rows is not None else " (no audio)")
    save_latent_cache_minimax(item, latent.cpu(), audio_latent=audio_rows,
                              still_latent=None if still_latent is None else still_latent.cpu(),
                              still_frame=still_frame)


# --- audio-only voice items --------------------------------------------------
def _is_audio(item: ItemInfo) -> bool:
    """Extension only — a voice item's content is a placeholder image the aggregator has
    already collapsed to a bare ndarray, so content shape can't distinguish it from a still."""
    from fizgig.minimax.audio import is_audio
    return is_audio(str(item.item_key))


def _encode_audio_item(item: ItemInfo, audio_vae) -> None:
    """One voice recording -> one cache file: a zeros video placeholder and the real audio rows.

    The zeros latent is not laziness — it is the dataset-mean in normalized latent space, and
    the trainer zeroes the video loss for these items, so its only jobs are sizing the packed
    sequence (latent_t drives the audio row count) and giving the audio rows something to
    attend across. The audio is the training item.
    """
    from fizgig.minimax.audio import grid_frames_for_samples, read_audio_file, trim_to_grid
    from fizgig.minimax.audio_vae import pack_audio
    from fizgig.minimax.model import audio_latents_for_frames, latent_frames_for_pixels

    path = str(item.item_key)
    wav = read_audio_file(path)                                   # (2, L) or AudioRejected
    frames = grid_frames_for_samples(wav.shape[1], os.path.basename(path))
    wav = trim_to_grid(wav, frames)                               # hop-exact sample count

    a_dev = next(audio_vae.parameters()).device
    a_dt = next(audio_vae.parameters()).dtype
    with torch.no_grad():
        z = audio_vae.encode(torch.from_numpy(wav).unsqueeze(0).to(a_dev, dtype=a_dt))
    audio_rows = pack_audio(z).float().cpu()

    want = audio_latents_for_frames(frames)
    got = audio_rows.shape[0] // 2
    if got != want:
        raise ValueError(
            f"{os.path.basename(path)}: audio encoded to {got} latents but the model wants "
            f"{want} — the hop-exact trim should make this impossible; please report it.")

    latent_t = latent_frames_for_pixels(frames)
    placeholder = torch.zeros(24, latent_t, item.bucket_size[1] // 16, item.bucket_size[0] // 16)
    logger.info("latent cache: %s -> audio %s + placeholder %s (video loss will be zeroed)",
                item.item_key, tuple(audio_rows.shape), tuple(placeholder.shape))
    save_latent_cache_minimax(item, placeholder, audio_latent=audio_rows, audio_only=True)


# --- batch encoders (called by the cache scripts) ----------------------------
def encode_and_save_latents(vae, batch: List[ItemInfo], audio_vae=None,
                            clip_still: bool = False) -> None:
    """Encode a batch through the H3 video VAE and save each 24-ch latent.

    A CLIP takes a different route: its content is the whole list of frames, encoded together so
    the VAE's causal 4x temporal stride produces real latent frames, and its audio (when it has
    any and was not muted) is encoded here too and stored in the same file. Clips are encoded one
    at a time — a batch mixing lengths cannot stack, and MiniMax trains at batch size 1 anyway.

    audio_vae=None means clips still train, silently: video learns, the audio rows stay silence.
    That is the intended behaviour when the user has not downloaded the audio VAE.
    """
    audio_items = [it for it in batch if _is_audio(it)]
    if audio_items:
        if len(batch) != len(audio_items):
            raise ValueError("a batch mixes voice recordings with other items — set Batch "
                             "Size to 1")
        if audio_vae is None:
            raise ValueError(
                "this dataset contains voice recordings but no audio VAE is configured — set "
                "the Audio VAE path in Preferences (Model Paths, MiniMax H3), or remove the "
                "audio files.")
        for item in audio_items:
            _encode_audio_item(item, audio_vae)
        return

    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype

    clips = [it for it in batch if _is_clip(it)]
    if clips:
        if len(batch) != len(clips):
            raise ValueError("a batch mixes clips and stills — MiniMax caches clips one at a "
                             "time; set Batch Size to 1")
        for item in clips:
            _encode_clip(vae, item, audio_vae, device, dtype, clip_still=clip_still)
        return

    contents = []
    for item in batch:
        content = item.content
        content = content[0] if isinstance(content, list) else content  # (H, W, C) uint8
        contents.append(torch.from_numpy(content))
    contents = torch.stack(contents, dim=0).permute(0, 3, 1, 2)  # (B, C, H, W)
    contents = contents / 127.5 - 1.0                            # -> [-1, 1]
    with torch.no_grad():
        latents = vae.encode(contents.to(device, dtype=dtype))  # (B, 24, 1, H/16, W/16)
    for b, item in enumerate(batch):
        latent = latents[b]                        # (24, 1, H/16, W/16)
        latent = latent.squeeze(1) if latent.dim() == 4 else latent  # -> (24, H/16, W/16)
        logger.info(f"latent cache: {item.item_key} -> {tuple(latent.shape)}")
        save_latent_cache_minimax(item, latent)


@torch.no_grad()
def encode_and_save_reference_text(encoder, batch: List[ItemInfo], reference_for) -> None:
    """Encode each item's caption against its assigned reference images, one file per slot.

    `reference_for(item)` returns [(slot, PIL image, latent, name), ...] — the pairing lives in
    the caching script, which is the only place that can see the whole dataset.

    One reference per encode rather than several packed together: the total cache is the same,
    but the teacher's sequence at training time stays short."""
    for item in batch:
        for slot, image, latent, name in reference_for(item):
            hidden, tags = encoder.encode_with_reference(item.caption, [image])
            save_reference_text_cache(item.text_encoder_output_cache_path, slot, hidden[0], tags,
                                      latent, caption=item.caption, reference=name)


def encode_and_save_text(encoder, batch: List[ItemInfo], batch_size: int = None) -> None:
    """Encode a caption batch through the Qwen3-VL-32B TE and save each (L, 5120) state.

    One forward for the whole batch rather than one per caption: the nvfp4-resident encoder
    dequantizes 351 weights per FORWARD, so this is where the caching pass spends its time.
    encode_batch right-pads, which is exactly equivalent for a causal stack (verified in
    tests/diag_batch_encode.py) and memoizes, so a repeated caption costs nothing.

    batch_size is the captions per FORWARD. Passing it through matters: encode_batch re-chunks
    internally at its own default of 8, so without this the caller's --batch_size only decided
    how many items arrived here, never how many went through the model at once."""
    kw = {} if batch_size is None else {"batch_size": int(batch_size)}
    embeds = encoder.encode_batch([item.caption for item in batch], **kw)
    for item, hidden in zip(batch, embeds):
        save_text_encoder_output_cache_minimax(item, hidden[0])
