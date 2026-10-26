"""Fibo 2.0's DiT as Fizgig trains it: diffusers' BriaFibo2Transformer2DModel with block swap added.

The family's checkpoint is diffusers' own class (fibo2/driver.load_dit builds it from its config and assigns the
tensors in), and that class has no block-swap hooks. The streaming calls have to sit *inside* the block loop: a
forward hook on a block would re-fire during a gradient-checkpointing recompute and park weights the backward
still needs - the "mat2 is on cpu" class of crash every family that swaps has guarded against, and the reason
enable_block_swap detaches a stale offloader before replacing it. So this subclass keeps diffusers' construction
and overrides only the forward, with the two stream calls around each block of the main stack.

The forked body is diffusers' text verbatim plus those calls: the parity guard builds a mini config on CPU and
asserts the two classes return bit-identical output, with and without grad enabled.
"""
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_bria_fibo2 import (
    BriaFibo2Transformer2DModel,
    Transformer2DModelOutput,
    compute_gist_count,
)

if TYPE_CHECKING:
    from fizgig.modules.offloading import ModelOffloader


class Fibo2DiT(BriaFibo2Transformer2DModel):
    """The Fibo 2.0 transformer plus Fizgig's block swap over its 44 main layers.

    The constructor mirrors diffusers' defaults so the class behaves as the base one for `from_config`; the
    family's loader passes every value explicitly anyway."""

    def __init__(self, in_channels: int = 32, patch_size: int = 2, dim: int = 3584, n_layers: int = 44,
                 n_refiner_layers: int = 2, n_heads: int = 28, norm_eps: float = 1e-5, cap_feat_dim: int = 7680,
                 injection_layer_ids: tuple[int, ...] = (6, 14, 22, 30, 38), perceiver_num_layers: int = 2,
                 min_num_gist_tokens: int = 256, max_num_gist_tokens: int = 512, gist_step: int = 64,
                 gist_min_text_len: int = 256, gist_max_text_len: int = 2048, rope_theta: float = 256.0,
                 t_scale: float = 1000.0, axes_dims: tuple[int, ...] = (32, 48, 48),
                 axes_lens: tuple[int, ...] = (2048, 512, 512)):
        super().__init__(
            in_channels=in_channels, patch_size=patch_size, dim=dim, n_layers=n_layers,
            n_refiner_layers=n_refiner_layers, n_heads=n_heads, norm_eps=norm_eps, cap_feat_dim=cap_feat_dim,
            injection_layer_ids=injection_layer_ids, perceiver_num_layers=perceiver_num_layers,
            min_num_gist_tokens=min_num_gist_tokens, max_num_gist_tokens=max_num_gist_tokens,
            gist_step=gist_step, gist_min_text_len=gist_min_text_len, gist_max_text_len=gist_max_text_len,
            rope_theta=rope_theta, t_scale=t_scale, axes_dims=axes_dims, axes_lens=axes_lens)
        self.blocks_to_swap = 0
        self.offloader: ModelOffloader | None = None

    # ---- block swap (Fizgig's shared offloader, as zimage and moss_image use it) ------------------------------
    def enable_block_swap(self, num_blocks, device, supports_backward: bool = True):
        """Park the last `num_blocks` of the 44 main layers on CPU and stream them just-in-time. The 2 noise
        refiners and the 2 perceiver layers stay resident - they are 4 blocks against 44."""
        from fizgig.modules.offloading import ModelOffloader
        if self.offloader is not None:
            self.offloader.remove_hooks()       # stale backward hooks double-swap blocks
        n = len(self.layers)
        if not 0 < num_blocks <= n - 2:
            raise ValueError(f"Fibo 2.0 block swap: 1..{n - 2} blocks, got {num_blocks}")
        self.blocks_to_swap = num_blocks
        self.offloader = ModelOffloader("fibo2", list(self.layers), n, num_blocks, supports_backward,
                                        torch.device(device))
        print(f"Fibo 2.0: block swap enabled. Swapping {num_blocks} of {n} layers to device {device}. "
              f"Supports backward: {supports_backward}")

    def move_to_device_except_swap_blocks(self, device):
        """Everything to `device` except the swapped layers' weights (the model is assumed to be on CPU)."""
        if self.blocks_to_swap:
            # try/finally: an OOM inside .to() must not leave the model holding the empty placeholder ModuleList
            # permanently - every later forward would then run with zero transformer blocks.
            layers = self.layers
            self.layers = nn.ModuleList()
            try:
                self.to(device)
            finally:
                self.layers = layers
        else:
            self.to(device)
        self.prepare_block_swap_before_forward()

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap:
            self.offloader.prepare_block_devices_before_forward(list(self.layers))

    def _log_swap_state(self, label):
        """One line per mode switch: how many streamed layers sit off the target device afterwards. A switch
        that strands parked blocks shows up in the list."""
        if not self.blocks_to_swap:
            return
        want = self.offloader.device.type
        off = [i for i, b in enumerate(self.layers) if next(b.parameters()).device.type != want]
        print(f"Fibo 2.0 block swap ({label}): {self.blocks_to_swap} parked, layers off-target: {off}")

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()
            self._log_swap_state("inference")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()
            self._log_swap_state("training")

    # ---- the forked forward (diffusers' body verbatim + the two stream calls per block) ------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        context_latents: list[torch.Tensor] | None = None,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor] | Transformer2DModelOutput:
        """diffusers' BriaFibo2Transformer2DModel.forward with block-streaming calls around the main layers.
        Keep in sync with `diffusers/models/transformers/transformer_bria_fibo2.py`."""
        batch_size, channels, height, width = hidden_states.shape
        patch_size = self.config.patch_size
        height_tokens, width_tokens = height // patch_size, width // patch_size
        num_image_tokens = height_tokens * width_tokens
        device = hidden_states.device

        # 1. Timestep embeddings: image tokens get the real timestep, clean tokens (the gist and the context) get t = 0
        adaln_noisy = self.t_embedder(timestep * self.config.t_scale)
        adaln_clean = self.t_embedder(torch.zeros_like(timestep))

        # 2. Patchify the image and the context images, which share the patch embedding. The image sits on RoPE plane
        # `max_num_gist_tokens + 1` whatever the gist count, and context image k on the k-th plane after it with its
        # own rows and columns
        image_plane = self.config.max_num_gist_tokens + 1
        tokens = [self._patchify(hidden_states)]
        ids = [self.create_coordinate_grid((1, height_tokens, width_tokens), (image_plane, 0, 0), device)]
        for k, latents in enumerate(context_latents or []):
            tokens.append(self._patchify(latents))
            context_size = (1, latents.shape[2] // patch_size, latents.shape[3] // patch_size)
            ids.append(self.create_coordinate_grid(context_size, (image_plane + k + 1, 0, 0), device))
        hidden_states = self.x_embedder(torch.cat(tokens, dim=1))
        image_freqs = self.rope_embedder(torch.cat([grid.flatten(0, 2) for grid in ids])).unsqueeze(0)

        # 3. Noise refiner. With context images, the noise mask marks the image tokens as the only noisy ones
        refiner_noise_mask = None
        if context_latents:
            refiner_noise_mask = torch.zeros(batch_size, hidden_states.shape[1], dtype=torch.long, device=device)
            refiner_noise_mask[:, :num_image_tokens] = 1
        for block in self.noise_refiner:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, None, image_freqs, adaln_noisy, refiner_noise_mask, adaln_noisy, adaln_clean
                )
            else:
                hidden_states = block(
                    hidden_states,
                    None,
                    image_freqs,
                    adaln_input=adaln_noisy,
                    noise_mask=refiner_noise_mask,
                    adaln_noisy=adaln_noisy,
                    adaln_clean=adaln_clean,
                )

        # 4. Gist count per prompt from its text length, then the Perceiver turns the first bundle into gist tokens.
        # In a batch, prompts with fewer gist tokens are padded to the largest count and masked out
        if encoder_attention_mask is not None:
            text_lengths = encoder_attention_mask.sum(dim=1).tolist()
        else:
            text_lengths = [encoder_hidden_states.shape[2]] * batch_size
        gist_counts = compute_gist_count(
            text_lengths,
            min_gist=self.config.min_num_gist_tokens,
            max_gist=self.config.max_num_gist_tokens,
            gist_step=self.config.gist_step,
            min_text_len=self.config.gist_min_text_len,
            max_text_len=self.config.gist_max_text_len,
        )
        num_gist_tokens = max(gist_counts)
        gist_mask = None
        if min(gist_counts) < num_gist_tokens:
            gist_mask = (
                torch.arange(num_gist_tokens, device=device) < torch.tensor(gist_counts, device=device)[:, None]
            )
        gist = self.perceiver(encoder_hidden_states[:, 0], num_gist_tokens, encoder_attention_mask, gist_mask)
        gist_ids = self.create_coordinate_grid((num_gist_tokens, 1, 1), (1, 0, 0), device).flatten(0, 2)

        # 5. One stream: gist tokens first, then the image tokens, then the context tokens, the order Fibo 2.0 was trained
        # with. When the batch pads gist tokens, the gist goes last instead, so the padding ends every row as varlen
        # attention backends expect. Positions come from RoPE, so the order changes nothing but float rounding. The
        # noise mask marks which tokens are noisy
        num_context_tokens = hidden_states.shape[1] - num_image_tokens
        gist_freqs = self.rope_embedder(gist_ids).unsqueeze(0)
        attention_mask = None
        if gist_mask is None:
            image_start = num_gist_tokens
            hidden_states = torch.cat([gist, hidden_states], dim=1)
            freqs_cis = torch.cat([gist_freqs, image_freqs], dim=1)
        else:
            image_start = 0
            hidden_states = torch.cat([hidden_states, gist], dim=1)
            freqs_cis = torch.cat([image_freqs, gist_freqs], dim=1)
            image_mask = torch.ones(batch_size, num_image_tokens + num_context_tokens, dtype=torch.bool, device=device)
            attention_mask = torch.cat([image_mask, gist_mask], dim=1)
        noise_mask = torch.zeros(batch_size, hidden_states.shape[1], dtype=torch.long, device=device)
        noise_mask[:, image_start : image_start + num_image_tokens] = 1

        # 6. The main blocks. Injection blocks also read their own bundle, in `injection_layer_ids` order
        bundle_index = {layer_id: index + 1 for index, layer_id in enumerate(self.config.injection_layer_ids)}
        for layer_id, block in enumerate(self.layers):
            text = encoder_hidden_states[:, bundle_index[layer_id]] if layer_id in bundle_index else None
            if self.blocks_to_swap:
                self.offloader.wait_for_block(layer_id)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    freqs_cis,
                    None,
                    noise_mask,
                    adaln_noisy,
                    adaln_clean,
                    text,
                    encoder_attention_mask,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask,
                    freqs_cis,
                    noise_mask=noise_mask,
                    adaln_noisy=adaln_noisy,
                    adaln_clean=adaln_clean,
                    encoder_hidden_states=text,
                    encoder_attention_mask=encoder_attention_mask,
                )
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.layers, layer_id)

        # 7. Final layer on the image tokens only, then unpatchify back to latents
        hidden_states = self.final_layer(hidden_states[:, image_start : image_start + num_image_tokens], c=adaln_noisy)
        hidden_states = hidden_states.view(batch_size, height_tokens, width_tokens, channels, patch_size, patch_size)
        output = hidden_states.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, channels, height, width)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
