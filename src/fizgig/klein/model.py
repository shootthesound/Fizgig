# Fizgig-native Klein DiT model
# Based on FLUX repo: https://github.com/black-forest-labs/flux
# License: Apache-2.0 License
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


from fizgig.modules.attention import AttentionParams
from fizgig.modules.offloading import ModelOffloader, weighs_to_device
from fizgig.modules.attention import attention as unified_attention


# region cpu offloading utilities


def _to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, list):
        return [_to_device(elem, device) for elem in x]
    elif isinstance(x, tuple):
        return tuple(_to_device(elem, device) for elem in x)
    elif isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    else:
        return x


def _to_cpu(x: Any) -> Any:
    """
    Recursively moves torch.Tensor objects (and containers thereof) to CPU.
    """
    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, list):
        return [_to_cpu(elem) for elem in x]
    elif isinstance(x, tuple):
        return tuple(_to_cpu(elem) for elem in x)
    elif isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    else:
        return x


def create_cpu_offloading_wrapper(func: Callable, device: torch.device) -> Callable:
    """
    Create a wrapper function that offloads inputs to CPU before calling the original function
    and moves outputs back to the specified device.
    """

    def wrapper(orig_func: Callable) -> Callable:
        def custom_forward(*inputs):
            nonlocal device, orig_func
            cuda_inputs = _to_device(inputs, device)
            outputs = orig_func(*cuda_inputs)
            return _to_cpu(outputs)

        return custom_forward

    return wrapper(func)


# endregion


FP8_OPTIMIZATION_TARGET_KEYS = ["double_blocks", "single_blocks"]
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["norm", "pe_embedder", "time_in", "_modulation"]


@dataclass
class Flux2Params:
    in_channels: int = 128  # packed latent channels
    context_in_dim: int = 15360
    hidden_size: int = 6144
    num_heads: int = 48
    depth: int = 8
    depth_single_blocks: int = 48
    axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 2000
    mlp_ratio: float = 3.0
    use_guidance_embed: bool = True


@dataclass
class Klein9BParams(Flux2Params):
    context_in_dim: int = 12288
    hidden_size: int = 4096
    num_heads: int = 32
    depth: int = 8
    depth_single_blocks: int = 24
    axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 2000
    mlp_ratio: float = 3.0
    use_guidance_embed: bool = False


# region autoencoder


@dataclass
class AutoEncoderParams:
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    z_channels: int = 32


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        from fizgig.modules.wide_attention import wide_head_attention
        h_ = wide_head_attention(q, k, v)            # one 512-wide head: computed directly on ROCm (#179)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.post_quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z)

        # get dtype for proper tracing
        upscale_dtype = next(self.up.parameters()).dtype

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # cast to proper dtype
        h = h.to(upscale_dtype)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.params = params
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )

        self.bn_eps = 1e-4
        self.bn_momentum = 0.1
        self.ps = [2, 2]
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.ps) * params.z_channels,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )

    def normalize(self, z):
        self.bn.eval()
        return self.bn(z)

    def inv_normalize(self, z):
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    def encode(self, x: Tensor) -> Tensor:
        moments = self.encoder(x)
        mean = torch.chunk(moments, 2, dim=1)[0]

        z = rearrange(
            mean,
            "... c (i pi) (j pj)  -> ... (c pi pj) i j",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        z = self.normalize(z)
        return z

    def decode(self, z: Tensor) -> Tensor:
        z = self.inv_normalize(z)
        z = rearrange(
            z,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        dec = self.decoder(z)
        return dec

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


# endregion

# region model


class KleinDiT(nn.Module):
    def __init__(self, params: Flux2Params, attn_mode: str = "flash", split_attn: bool = False) -> None:
        super().__init__()

        self.in_channels = params.in_channels
        self.out_channels = params.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}")
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads

        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=False)

        self.use_guidance_embed = params.use_guidance_embed
        if self.use_guidance_embed:
            self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)

        self.attn_mode = attn_mode
        self.split_attn = split_attn

        self.double_blocks = nn.ModuleList(
            [DoubleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio) for _ in range(params.depth)]
        )
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.double_stream_modulation_img = Modulation(self.hidden_size, double=True, disable_bias=True)
        self.double_stream_modulation_txt = Modulation(self.hidden_size, double=True, disable_bias=True)
        self.single_stream_modulation = Modulation(self.hidden_size, double=False, disable_bias=True)

        self.final_layer = LastLayer(self.hidden_size, self.out_channels)

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        self.blocks_to_swap = None

        self.offloader_double = None
        self.offloader_single = None
        self.num_double_blocks = len(self.double_blocks)
        self.num_single_blocks = len(self.single_blocks)

    def get_model_type(self) -> str:
        return "flux_2"

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading

        self.time_in.enable_gradient_checkpointing()
        if self.use_guidance_embed and self.guidance_in.__class__ != nn.Identity:
            self.guidance_in.enable_gradient_checkpointing()

        for block in self.double_blocks + self.single_blocks:
            block.enable_gradient_checkpointing(activation_cpu_offloading)

        print(f"KleinDiT: Gradient checkpointing enabled. Activation CPU offloading: {activation_cpu_offloading}")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

        self.time_in.disable_gradient_checkpointing()
        if self.use_guidance_embed and self.guidance_in.__class__ != nn.Identity:
            self.guidance_in.disable_gradient_checkpointing()

        for block in self.double_blocks + self.single_blocks:
            block.disable_gradient_checkpointing()

        print("KleinDiT: Gradient checkpointing disabled.")

    def enable_block_swap(self, num_blocks: int, device: torch.device, supports_backward: bool, use_pinned_memory: bool = False,
                          double_blocks_to_swap=None, single_blocks_to_swap=None):
        # Detach any previous offloaders' backward hooks before replacing them.
        # enable_block_swap is called repeatedly during training (e.g. the Distilled
        # sample path maxes swap to free VRAM, then restores it), and each call used
        # to leave the old offloaders' register_full_backward_hook handles attached
        # until GC. Stale hooks fire alongside the new ones on the next backward,
        # double-swapping blocks and stranding base weights on CPU -> "mat2 is on
        # cpu" crash / corrupted gradients on the first post-sample step.
        for _off in (getattr(self, "offloader_double", None), getattr(self, "offloader_single", None)):
            if _off is not None and hasattr(_off, "remove_hooks"):
                _off.remove_hooks()

        # Explicit per-type override: bypass the ratio formula and swap exactly the
        # requested double/single counts. The formula locks single = 3x double, so
        # it tops out at 24 swapped (2 double + 6 single resident) for Klein 9B; the
        # Distilled sample path uses this override to reach the real max of 28
        # (2 double + 2 single resident), freeing ~1 GB more for the second model.
        if double_blocks_to_swap is not None and single_blocks_to_swap is not None:
            double_blocks_to_swap = int(double_blocks_to_swap)
            single_blocks_to_swap = int(single_blocks_to_swap)
            self.blocks_to_swap = double_blocks_to_swap + single_blocks_to_swap
        else:
            self.blocks_to_swap = num_blocks
            if num_blocks <= 0:
                double_blocks_to_swap = 0
                single_blocks_to_swap = 0
            elif self.num_double_blocks == 0:
                double_blocks_to_swap = 0
                single_blocks_to_swap = num_blocks
            elif self.num_single_blocks == 0:
                double_blocks_to_swap = num_blocks
                single_blocks_to_swap = 0
            else:
                swap_ratio = self.num_single_blocks / self.num_double_blocks
                double_blocks_to_swap = int(round(num_blocks / (1.0 + swap_ratio / 2.0)))
                single_blocks_to_swap = int(round(double_blocks_to_swap * swap_ratio))

                # adjust if we exceed available blocks
                if self.num_double_blocks * 2 < self.num_single_blocks:
                    while double_blocks_to_swap >= 1 and double_blocks_to_swap > self.num_double_blocks - 2:
                        double_blocks_to_swap -= 1
                        single_blocks_to_swap += 2
                else:
                    while single_blocks_to_swap >= 2 and single_blocks_to_swap > self.num_single_blocks - 2:
                        single_blocks_to_swap -= 2
                        double_blocks_to_swap += 1

                if double_blocks_to_swap == 0 and single_blocks_to_swap == 0:
                    if self.num_single_blocks >= self.num_double_blocks:
                        single_blocks_to_swap = 1
                    else:
                        double_blocks_to_swap = 1

        assert double_blocks_to_swap <= self.num_double_blocks - 2 and single_blocks_to_swap <= self.num_single_blocks - 2, (
            f"Cannot swap more than {self.num_double_blocks - 2} double blocks and {self.num_single_blocks - 2} single blocks. "
            f"Requested {double_blocks_to_swap} double blocks and {single_blocks_to_swap} single blocks."
        )

        self.offloader_double = ModelOffloader(
            "double",
            self.double_blocks,
            self.num_double_blocks,
            double_blocks_to_swap,
            supports_backward,
            device,
            use_pinned_memory,
        )
        self.offloader_single = ModelOffloader(
            "single",
            self.single_blocks,
            self.num_single_blocks,
            single_blocks_to_swap,
            supports_backward,
            device,
            use_pinned_memory,
        )
        print(
            f"KleinDiT: Block swap enabled. Swapping {num_blocks} blocks, double blocks: {double_blocks_to_swap}, single blocks: {single_blocks_to_swap}."
        )

    def disable_block_swap(self):
        """Fully tear down block swap: detach offloader backward hooks and drop the
        offloaders. Use when returning to no-swap training (e.g. the Distilled
        sample path maxes swap to free VRAM, then restores swap=0). Without this,
        the sample-time offloaders' register_full_backward_hook handles linger and
        fire on the next training backward — perturbing the gradient-checkpointing
        recompute (harmless to plain Linear, but it broke the fp8 scaled_mm path)."""
        for attr in ("offloader_double", "offloader_single"):
            off = getattr(self, attr, None)
            if off is not None and hasattr(off, "remove_hooks"):
                off.remove_hooks()
            setattr(self, attr, None)
        self.blocks_to_swap = 0

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader_double.set_forward_only(True)
            self.offloader_single.set_forward_only(True)
            self.prepare_block_swap_before_forward()
            print("KleinDiT: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader_double.set_forward_only(False)
            self.offloader_single.set_forward_only(False)
            self.prepare_block_swap_before_forward()
            print("KleinDiT: Block swap set to forward and backward.")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        # assume model is on cpu. do not move blocks to device to reduce temporary memory usage
        # try/finally: an OOM inside .to() must not leave the model holding the empty
        # placeholder ModuleLists permanently (a silent lobotomy — every later forward
        # would run with zero transformer blocks).
        if self.blocks_to_swap:
            save_double_blocks = self.double_blocks
            save_single_blocks = self.single_blocks
            self.double_blocks = nn.ModuleList()
            self.single_blocks = nn.ModuleList()

        try:
            self.to(device)
        finally:
            if self.blocks_to_swap:
                self.double_blocks = save_double_blocks
                self.single_blocks = save_single_blocks

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader_double.prepare_block_devices_before_forward(self.double_blocks)
        self.offloader_single.prepare_block_devices_before_forward(self.single_blocks)

    def forward(self, x: Tensor, x_ids: Tensor, timesteps: Tensor, ctx: Tensor, ctx_ids: Tensor, guidance: Tensor | None) -> Tensor:
        num_txt_tokens = ctx.shape[1]

        timestep_emb = timestep_embedding(timesteps, 256)
        del timesteps
        vec = self.time_in(timestep_emb)
        del timestep_emb
        if self.use_guidance_embed:
            guidance_emb = timestep_embedding(guidance, 256)
            vec = vec + self.guidance_in(guidance_emb)
            del guidance_emb

        double_block_mod_img = self.double_stream_modulation_img(vec)
        double_block_mod_txt = self.double_stream_modulation_txt(vec)
        single_block_mod, _ = self.single_stream_modulation(vec)

        img = self.img_in(x)
        del x
        txt = self.txt_in(ctx)
        del ctx
        pe_x = self.pe_embedder(x_ids)
        del x_ids
        pe_ctx = self.pe_embedder(ctx_ids)
        del ctx_ids

        attn_params = AttentionParams.create(self.attn_mode, self.split_attn)  # No attention mask

        for block_idx, block in enumerate(self.double_blocks):
            if self.blocks_to_swap:
                self.offloader_double.wait_for_block(block_idx)

            img, txt = block(img, txt, pe_x, pe_ctx, double_block_mod_img, double_block_mod_txt, attn_params)

            if self.blocks_to_swap:
                self.offloader_double.submit_move_blocks_forward(self.double_blocks, block_idx)

        del double_block_mod_img, double_block_mod_txt

        img = torch.cat((txt, img), dim=1)
        del txt
        pe = torch.cat((pe_ctx, pe_x), dim=2)
        del pe_ctx, pe_x

        for block_idx, block in enumerate(self.single_blocks):
            if self.blocks_to_swap:
                self.offloader_single.wait_for_block(block_idx)

            img = block(img, pe, single_block_mod, attn_params)

            if self.blocks_to_swap:
                self.offloader_single.submit_move_blocks_forward(self.single_blocks, block_idx)

        del single_block_mod, pe

        img = img.to(vec.device)  # move to gpu if gradient checkpointing cpu offloading is used

        img = img[:, num_txt_tokens:, ...]

        img = self.final_layer(img, vec)
        return img


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)

        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim, bias=False)


class SiLUActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool, disable_bias: bool = False):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=not disable_bias)

    def forward(self, vec: torch.Tensor):
        org_dtype = vec.dtype
        vec = vec.to(torch.float32)  # for numerical stability
        out = self.lin(nn.functional.silu(vec))
        if out.ndim == 2:
            out = out[:, None, :]
        out = out.to(org_dtype)
        out = out.chunk(self.multiplier, dim=-1)
        return out[:3], out[3:] if self.is_double else None


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        org_dtype = x.dtype
        vec = vec.to(torch.float32)  # for numerical stability
        mod = self.adaLN_modulation(vec)
        shift, scale = mod.chunk(2, dim=-1)
        if shift.ndim == 2:
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        x = x.to(torch.float32)  # for numerical stability
        x = (1 + scale) * self.norm_final(x) + shift
        x = self.linear(x)
        return x.to(org_dtype)


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()

        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = head_dim**-0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_mult_factor = 2

        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim * self.mlp_mult_factor, bias=False)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=False)

        self.norm = QKNorm(head_dim)
        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = SiLUActivation()

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def _forward(self, x: Tensor, pe: Tensor, mod: tuple[Tensor, Tensor], attn_params: AttentionParams) -> Tensor:
        mod_shift, mod_scale, mod_gate = mod
        del mod
        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift
        del mod_scale, mod_shift

        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim * self.mlp_mult_factor], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        del qkv
        q, k = self.norm(q, k, v)

        qkv_list = [q, k, v]
        del q, k, v
        attn = attention(qkv_list, pe, attn_params)
        del qkv_list, pe

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod_gate * output

    def forward(self, x: Tensor, pe: Tensor, mod: tuple[Tensor, Tensor], attn_params: AttentionParams) -> Tensor:
        if self.training and self.gradient_checkpointing:
            forward_fn = self._forward
            if self.activation_cpu_offloading:
                forward_fn = create_cpu_offloading_wrapper(forward_fn, self.linear1.weight.device)
            return checkpoint(forward_fn, x, pe, mod, attn_params, use_reentrant=False)
        else:
            return self._forward(x, pe, mod, attn_params)


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0, f"{hidden_size=} must be divisible by {num_heads=}"

        self.hidden_size = hidden_size
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_mult_factor = 2

        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * self.mlp_mult_factor, bias=False),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )

        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * self.mlp_mult_factor, bias=False),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def _forward(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        pe_ctx: Tensor,
        mod_img: tuple[Tensor, Tensor],
        mod_txt: tuple[Tensor, Tensor],
        attn_params: AttentionParams,
    ) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = mod_img
        txt_mod1, txt_mod2 = mod_txt
        del mod_img, mod_txt

        img_mod1_shift, img_mod1_scale, img_mod1_gate = img_mod1
        img_mod2_shift, img_mod2_scale, img_mod2_gate = img_mod2
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate = txt_mod1
        txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_mod2
        del img_mod1, img_mod2, txt_mod1, txt_mod2

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1_scale) * img_modulated + img_mod1_shift
        del img_mod1_scale, img_mod1_shift

        img_qkv = self.img_attn.qkv(img_modulated)
        del img_modulated
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        del img_qkv
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1_scale) * txt_modulated + txt_mod1_shift
        del txt_mod1_scale, txt_mod1_shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        del txt_modulated
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        del txt_qkv
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        txt_len = txt_q.shape[2]
        q = torch.cat((txt_q, img_q), dim=2)
        del txt_q, img_q
        k = torch.cat((txt_k, img_k), dim=2)
        del txt_k, img_k
        v = torch.cat((txt_v, img_v), dim=2)
        del txt_v, img_v

        pe = torch.cat((pe_ctx, pe), dim=2)
        del pe_ctx
        qkv_list = [q, k, v]
        del q, k, v
        attn = attention(qkv_list, pe, attn_params)
        del qkv_list, pe
        txt_attn, img_attn = attn[:, :txt_len], attn[:, txt_len:]
        del attn

        # calculate the img blocks
        img = img + img_mod1_gate * self.img_attn.proj(img_attn)
        del img_mod1_gate, img_attn
        img = img + img_mod2_gate * self.img_mlp((1 + img_mod2_scale) * (self.img_norm2(img)) + img_mod2_shift)
        del img_mod2_gate, img_mod2_scale, img_mod2_shift

        # calculate the txt blocks
        txt = txt + txt_mod1_gate * self.txt_attn.proj(txt_attn)
        del txt_mod1_gate, txt_attn
        txt = txt + txt_mod2_gate * self.txt_mlp((1 + txt_mod2_scale) * (self.txt_norm2(txt)) + txt_mod2_shift)
        del txt_mod2_gate, txt_mod2_scale, txt_mod2_shift
        return img, txt

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        pe_ctx: Tensor,
        mod_img: tuple[Tensor, Tensor],
        mod_txt: tuple[Tensor, Tensor],
        attn_params: AttentionParams,
    ) -> tuple[Tensor, Tensor]:
        if self.training and self.gradient_checkpointing:
            forward_fn = self._forward
            if self.activation_cpu_offloading:
                forward_fn = create_cpu_offloading_wrapper(forward_fn, self.img_mlp[0].weight.device)
            return checkpoint(forward_fn, img, txt, pe, pe_ctx, mod_img, mod_txt, attn_params, use_reentrant=False)
        else:
            return self._forward(img, txt, pe, pe_ctx, mod_img, mod_txt, attn_params)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, disable_bias: bool = False):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=not disable_bias)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=not disable_bias)
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def _forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))

    def forward(self, *args, **kwargs):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, *args, use_reentrant=False, **kwargs)
        else:
            return self._forward(*args, **kwargs)


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(len(self.axes_dim))], dim=-3)
        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, device=t.device, dtype=torch.float32) / half)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


def attention(qkv_list: list[Tensor], pe: Tensor, attn_params: AttentionParams) -> Tensor:
    q, k, v = qkv_list
    del qkv_list
    q, k = apply_rope(q, k, pe)

    q = q.transpose(1, 2)  # B, H, L, D -> B, L, H, D
    k = k.transpose(1, 2)  # B, H, L, D -> B, L, H, D
    v = v.transpose(1, 2)  # B, H, L, D -> B, L, H, D
    qkv_list = [q, k, v]
    del q, k, v
    x = unified_attention(qkv_list, attn_params=attn_params)
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


# endregion
