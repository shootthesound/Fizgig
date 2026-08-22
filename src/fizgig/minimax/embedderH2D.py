from __future__ import annotations

import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen


# ============================================================================
# QWEN CONFIG
# ============================================================================

_QWEN3_32B_TRUNC50 = dict(
    hidden_size=5120,
    num_hidden_layers=50,
    num_attention_heads=64,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=25600,
    vocab_size=151936,
    max_position_embeddings=262144,
    rms_norm_eps=1e-6,
    rope_theta=5000000.0,
    attention_bias=False,
    tie_word_embeddings=False,
)


_NF4_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)


_E2M1_MAG = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
    ],
    dtype=torch.float32,
)

_E2M1_SIGNED = torch.cat(
    [
        _E2M1_MAG,
        -_E2M1_MAG,
    ]
)


# ============================================================================
# NVFP4
# ============================================================================

def _from_blocked(
    blocked,
    rows,
    cols,
):
    nrb = -(-rows // 128)
    ncb = -(-cols // 4)

    x = blocked.reshape(
        -1,
        32,
        4,
        4,
    ).transpose(
        1,
        2,
    )

    x = x.reshape(
        nrb,
        ncb,
        128,
        4,
    )

    x = x.permute(
        0,
        2,
        1,
        3,
    )

    x = x.reshape(
        nrb * 128,
        ncb * 4,
    )

    return x[
        :rows,
        :cols,
    ].contiguous()


def _nvfp4_dequant(packed, block_scale_fp8, global_scale):
    """packed U8 [out, in/2] -> bf16 [out, in]."""
    out, in2 = packed.shape
    inp = in2 * 2
    dev = packed.device

    bs = _from_blocked(
        block_scale_fp8.to(torch.float32).reshape(-1, 32, 16),
        out,
        inp // 16,
    )

    gs = global_scale.to(torch.float32)
    table = _E2M1_SIGNED.to(dev)

    w_out = torch.empty(
        out,
        inp,
        dtype=torch.bfloat16,
        device=dev,
    )

    chunk = max(
        1,
        (32 << 20) // max(inp, 1),
    )

    for r0 in range(
        0,
        out,
        chunk,
    ):

        r1 = min(
            out,
            r0 + chunk,
        )

        codes = torch.empty(
            r1 - r0,
            inp,
            dtype=torch.uint8,
            device=dev,
        )

        codes[:, 0::2] = (
            packed[r0:r1] >> 4
        )

        codes[:, 1::2] = (
            packed[r0:r1] & 0x0F
        )

        vals = table[
            codes.long()
        ]

        w = (
            vals.view(
                r1 - r0,
                inp // 16,
                16,
            )
            * bs[r0:r1].unsqueeze(-1)
        ).mul_(gs)

        w_out[
            r0:r1
        ] = w.view(
            r1 - r0,
            inp,
        ).to(
            torch.bfloat16
        )

    return w_out


def _dequant_comfy_weight(
    f,
    file_mod,
    ckpt,
):
    fmt = ""

    try:

        blob = bytes(
            f.get_tensor(
                file_mod
                + ".comfy_quant"
            ).tolist()
        )

        fmt = json.loads(
            blob.decode("utf-8")
        ).get(
            "format",
            "",
        )

    except Exception:
        pass

    if fmt not in (
        "nvfp4",
        "int8_tensorwise",
    ):

        raise NotImplementedError(
            f"Unsupported comfy-quant format "
            f"'{fmt or 'unknown'}' on {file_mod}."
        )

    if fmt == "nvfp4":

        packed = f.get_tensor(
            file_mod + ".weight"
        )

        bscale = f.get_tensor(
            file_mod + ".weight_scale"
        )

        gscale = f.get_tensor(
            file_mod + ".weight_scale_2"
        )

        w = _nvfp4_dequant(
            packed,
            bscale,
            gscale,
        )

        pqs_key = (
            file_mod
            + ".pre_quant_scale"
        )

        if pqs_key in ckpt:

            pqs = f.get_tensor(
                pqs_key
            ).to(torch.float32)

            w = (
                w.to(torch.float32)
                * pqs.unsqueeze(0)
            ).to(
                torch.bfloat16
            )

        return w

    w = f.get_tensor(
        file_mod + ".weight"
    ).to(torch.float32)

    s = f.get_tensor(
        file_mod + ".weight_scale"
    ).to(torch.float32)

    return (
        w * s
    ).to(
        torch.bfloat16
    )


# ============================================================================
# TOKENIZER
# ============================================================================

def _bundled_tokenizer_dir():

    return os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "..",
        "assets",
        "qwen3vl_tokenizer",
    )


_H3_SPECIAL_TOKENS = (
    "<d>",
    "</d>",
    "<|cutoff|>",
    "<|lyrics_start|>",
    "<|lyrics_end|>",
    "<|caption_start|>",
    "<|caption_end|>",
)


def _add_h3_special_tokens(
    tok,
):

    missing = [
        t
        for t in _H3_SPECIAL_TOKENS
        if tok.convert_tokens_to_ids(t) is None
    ]

    if missing:

        tok.add_tokens(
            missing,
            special_tokens=True,
        )


# ============================================================================
# MODEL BUILDERS
# ============================================================================

def build_qwen3_te(
    config_overrides=None,
):

    from transformers import (
        Qwen3Config,
        Qwen3Model,
    )

    cfg = dict(
        _QWEN3_32B_TRUNC50
    )

    if config_overrides:
        cfg.update(
            config_overrides
        )

    model = Qwen3Model(
        Qwen3Config(
            **cfg
        )
    )

    model.norm = nn.Identity()

    return model


_QWEN3VL_VISION = dict(
    out_hidden_size=5120,
)


_QWEN3VL_ROPE_SCALING = {
    "rope_type": "default",
    "mrope_section": [
        24,
        20,
        20,
    ],
    "mrope_interleaved": True,
}


def build_qwen3vl_te(
    config_overrides=None,
):

    from transformers import (
        Qwen3VLConfig,
        Qwen3VLModel,
    )

    txt = dict(
        _QWEN3_32B_TRUNC50
    )

    txt["rope_scaling"] = dict(
        _QWEN3VL_ROPE_SCALING
    )

    vis = dict(
        _QWEN3VL_VISION
    )

    if config_overrides:

        txt.update(
            config_overrides.get(
                "text",
                {},
            )
        )

        vis.update(
            config_overrides.get(
                "vision",
                {},
            )
        )

    model = Qwen3VLModel(
        Qwen3VLConfig(
            text_config=txt,
            vision_config=vis,
        )
    )

    model.language_model.norm = nn.Identity()

    return model


# ============================================================================
# REFERENCE SUPPORT
# ============================================================================

VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2


def build_image_processor(
    tokenizer_dir=None,
):

    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(
        tokenizer_dir
        or _bundled_tokenizer_dir()
    )


def build_reference_tokens(
    tokenizer,
    image_processor,
    caption,
    images,
    max_length=None,
):

    pixel_values = None
    image_grid_thw = None

    ids = []
    tags = []

    if images:

        vision = image_processor(
            images=images,
            return_tensors="pt",
        )

        pixel_values = vision[
            "pixel_values"
        ]

        image_grid_thw = vision[
            "image_grid_thw"
        ]

        merge = (
            image_processor.merge_size ** 2
        )

        v_start = tokenizer.convert_tokens_to_ids(
            "<|vision_start|>"
        )

        v_end = tokenizer.convert_tokens_to_ids(
            "<|vision_end|>"
        )

        img_pad = tokenizer.convert_tokens_to_ids(
            "<|image_pad|>"
        )

        for i in range(
            len(images)
        ):

            n_img = (
                int(
                    image_grid_thw[i].prod()
                )
                // merge
            )

            label = tokenizer(
                f"<Picture {i + 1}>: ",
                add_special_tokens=False,
            )[
                "input_ids"
            ]

            vision_ids = (
                [v_start]
                + [img_pad] * n_img
                + [v_end]
            )

            ids += label
            ids += vision_ids

            tags += (
                [TEXT_TAG]
                * len(label)
            )

            tags += (
                [VIDEO_TAG]
                * len(vision_ids)
            )

    _tk = dict(
        add_special_tokens=False
    )

    if max_length:

        _tk.update(
            truncation=True,
            max_length=max_length,
        )

    prompt_ids = tokenizer(
        caption,
        **_tk,
    )[
        "input_ids"
    ]

    ids += prompt_ids

    tags += (
        [TEXT_TAG]
        * len(prompt_ids)
    )

    if not ids:

        pid = getattr(
            tokenizer,
            "pad_token_id",
            None,
        )

        pid = (
            151643
            if pid is None
            else int(pid)
        )

        ids = [pid]
        tags = [TEXT_TAG]

    return (
        torch.tensor(
            [ids],
            dtype=torch.long,
        ),
        torch.tensor(
            tags,
            dtype=torch.long,
        ),
        pixel_values,
        image_grid_thw,
    )


# ============================================================================
# NVFP4 LINEAR
# ============================================================================

class Nvfp4Linear(nn.Linear):

    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        compute_dtype=torch.bfloat16,
    ):

        super().__init__(
            in_features,
            out_features,
            bias=bias,
        )

        del self._parameters["weight"]

        self.compute_dtype = (
            compute_dtype
        )

        # --------------------------------------------------------------
        # CPU-only NVFP4 source buffers.
        #
        # These are registered as buffers so they remain part of the
        # module, but _apply() below deliberately protects them from
        # .to(cuda) / .to(cpu).
        # --------------------------------------------------------------

        self.register_buffer(
            "packed",
            torch.empty(
                out_features,
                in_features // 2,
                dtype=torch.uint8,
                device="cpu",
            ),
            persistent=False,
        )

        self.register_buffer(
            "bscale",
            torch.empty(
                out_features,
                in_features // 16,
                dtype=torch.uint8,
                device="cpu",
            ),
            persistent=False,
        )

        self.register_buffer(
            "gscale",
            torch.empty(
                4,
                dtype=torch.uint8,
                device="cpu",
            ),
            persistent=False,
        )

        self.register_buffer(
            "pre_quant_scale",
            None,
            persistent=False,
        )

        # --------------------------------------------------------------
        # GPU ring references.
        # --------------------------------------------------------------

        self._gpu_packed = None
        self._gpu_bscale = None
        self._gpu_gscale = None
        self._gpu_pqs = None
        self._gpu_bias = None

    # ----------------------------------------------------------------
    # IMPORTANT:
    #
    # nn.Module._apply() normally walks through _buffers and
    # _parameters and applies the device conversion to them.
    #
    # We MUST prevent that for:
    #
    #   packed
    #   bscale
    #   gscale
    #   pre_quant_scale
    #   bias
    #
    # Otherwise:
    #
    #   layer.to(cuda)
    #
    # would perform a full H2D copy of the quant payload before our
    # ring streamer performs its own H2D copy.
    #
    # The ring is the ONLY mechanism allowed to put these tensors
    # on GPU.
    # ----------------------------------------------------------------

    def _apply(
        self,
        fn,
        recurse=True,
    ):
        packed = self.packed
        bscale = self.bscale
        gscale = self.gscale
        pqs = self.pre_quant_scale

        bias = self.bias

        # ----------------------------------------------------------
        # Temporarily unregister quant buffers.
        # ----------------------------------------------------------

        self._buffers.pop(
            "packed",
            None,
        )

        self._buffers.pop(
            "bscale",
            None,
        )

        self._buffers.pop(
            "gscale",
            None,
        )

        self._buffers.pop(
            "pre_quant_scale",
            None,
        )

        # ----------------------------------------------------------
        # Temporarily unregister bias.
        #
        # bias is an nn.Parameter, not a buffer.
        # ----------------------------------------------------------

        self._parameters.pop(
            "bias",
            None,
        )

        try:

            super()._apply(
                fn,
                recurse,
            )

        finally:

            # ------------------------------------------------------
            # Restore original CPU tensors.
            # ------------------------------------------------------

            self._buffers[
                "packed"
            ] = packed

            self._buffers[
                "bscale"
            ] = bscale

            self._buffers[
                "gscale"
            ] = gscale

            self._buffers[
                "pre_quant_scale"
            ] = pqs

            self._parameters[
                "bias"
            ] = bias

        return self

    # ----------------------------------------------------------------
    # Legacy direct GPU loader.
    #
    # Qwen3VLLayerStreamer does NOT use this for streaming.
    # It remains here for compatibility.
    # ----------------------------------------------------------------

    def load_to_gpu(
        self,
        device,
    ):

        self._gpu_packed = self.packed.to(
            device,
            non_blocking=True,
        )

        self._gpu_bscale = self.bscale.to(
            device,
            non_blocking=True,
        )

        self._gpu_gscale = self.gscale.to(
            device,
            non_blocking=True,
        )

        if self.pre_quant_scale is not None:

            self._gpu_pqs = (
                self.pre_quant_scale.to(
                    device,
                    non_blocking=True,
                )
            )

        else:

            self._gpu_pqs = None

        if self.bias is not None:

            self._gpu_bias = (
                self.bias.to(
                    device,
                    non_blocking=True,
                )
            )

    # ----------------------------------------------------------------
    # Clear GPU ring references.
    # ----------------------------------------------------------------

    def unload_gpu(self):

        self._gpu_packed = None
        self._gpu_bscale = None
        self._gpu_gscale = None
        self._gpu_pqs = None
        self._gpu_bias = None

    # ----------------------------------------------------------------
    # Scale conversion
    # ----------------------------------------------------------------

    def _scales(
        self,
        packed,
        gscale,
    ):

        return (
            packed.view(
                torch.float8_e4m3fn
            ),
            gscale.view(
                torch.float32
            ),
        )

    # ----------------------------------------------------------------
    # Weight property
    #
    # Used only when explicitly requested.
    # Normal streaming forward uses the GPU ring references directly.
    # ----------------------------------------------------------------

    @property
    def weight(self):

        if self._gpu_packed is not None:

            packed = self._gpu_packed
            bscale = self._gpu_bscale
            gscale = self._gpu_gscale
            pqs = self._gpu_pqs

        else:

            packed = self.packed
            bscale = self.bscale
            gscale = self.gscale
            pqs = self.pre_quant_scale

        bs, gs = self._scales(
            bscale,
            gscale,
        )

        w = _nvfp4_dequant(
            packed,
            bs,
            gs,
        ).to(
            self.compute_dtype
        )

        if pqs is not None:

            w = (
                w
                * pqs.to(
                    w.dtype
                ).reshape(
                    1,
                    -1,
                )
            )

        return w

    # ----------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------

    def forward(
        self,
        x,
    ):

        if self._gpu_packed is None:

            raise RuntimeError(
                "Nvfp4Linear GPU weights "
                "are not loaded."
            )

        dt = self.compute_dtype

        pqs = self._gpu_pqs

        if pqs is not None:

            x = (
                x
                * pqs.to(
                    x.dtype
                )
            )

        bs, gs = self._scales(
            self._gpu_bscale,
            self._gpu_gscale,
        )

        w = _nvfp4_dequant(
            self._gpu_packed,
            bs,
            gs,
        ).to(dt)

        bias = self._gpu_bias

        # Compatibility fallback.
        #
        # In the H2D streamer bias should always be present in the
        # ring slot when the Linear has a bias. This fallback keeps
        # the module usable with the legacy direct loader.
        if (
            bias is None
            and self.bias is not None
        ):

            bias = self.bias

        return F.linear(
            x.to(dt),
            w,
            bias,
        )


# ============================================================================
# 2 GPU SLOT H2D STREAMER
# ============================================================================

class Qwen3VLLayerStreamer:
    """
    H2D-only streaming для Qwen3-VL-32B NVFP4.

    На GPU постоянно существуют только 3 layer slots.

        slot 0 -> один decoder layer
        slot 1 -> один decoder layer

    Например:

        compute layer 0
             +
        H2D layer 2

        compute layer 1
             +
        H2D layer 3

        compute layer 2
             +
        H2D layer 4

    В каждом slot находятся только NVFP4 buffers
    ОДНОГО слоя.

    CPU:
        все packed NVFP4 веса всех 50 слоёв

    GPU:
        максимум 2 набора NVFP4 весов
        + параметры norm/attention самого активного слоя
        + embeddings / rotary / служебные тензоры

    Quant payload и bias загружаются в GPU ТОЛЬКО через ring.
    """

    def __init__(
        self,
        model,
        device="cuda",
        ring_size=2,
    ):

        self.model = model
        self.device = torch.device(device)

        if hasattr(model, "language_model"):
            self.layers = list(
                model.language_model.layers
            )
        else:
            self.layers = list(
                model.layers
            )

        self.n_layers = len(self.layers)

        self.ring_size = min(
            max(1, int(ring_size)),
            self.n_layers,
        )

        self.copy_stream = torch.cuda.Stream(
            device=self.device
        )

        # layer -> H2D completion event
        self.copy_done = [
            None
        ] * self.n_layers

        # slot -> event saying that GPU finished using the slot
        self.free_event = [
            None
        ] * self.ring_size

        # slot -> currently loaded layer
        self.loaded_layer = [
            None
        ] * self.ring_size
        # ------------------------------------------------------------
        # Each slot contains buffers for ONE layer only.
        #
        # IMPORTANT:
        # Do NOT create buffers for all 50 layers here.
        # ------------------------------------------------------------

        self.slots = [
            None
            for _ in range(self.ring_size)
        ]

        # Number / order of NVFP4 linears in a decoder layer.
        self.template_linears = self._nvfp4_linears(
            self.layers[0]
        )
  
        self._allocate_slots()

    # ================================================================
    # NVFP4 discovery
    # ================================================================

    @staticmethod
    def _nvfp4_linears(layer):

        return [
            m
            for m in layer.modules()
            if isinstance(
                m,
                Nvfp4Linear,
            )
        ]

    # ================================================================
    # GPU allocation
    # ================================================================

    def _allocate_slots(self):

        print(
            f"[Qwen3 H2D] allocating "
            f"{self.ring_size} GPU layer slots...",
            flush=True,
        )

        for slot_idx in range(
            self.ring_size
        ):

            buffers = []

            for linear in self.template_linears:

                buffers.append(
                    {
                        "packed": torch.empty(
                            linear.packed.shape,
                            dtype=linear.packed.dtype,
                            device=self.device,
                        ),

                        "bscale": torch.empty(
                            linear.bscale.shape,
                            dtype=linear.bscale.dtype,
                            device=self.device,
                        ),

                        "gscale": torch.empty(
                            linear.gscale.shape,
                            dtype=linear.gscale.dtype,
                            device=self.device,
                        ),

                        "pqs": (
                            torch.empty(
                                linear.pre_quant_scale.shape,
                                dtype=linear.pre_quant_scale.dtype,
                                device=self.device,
                            )
                            if linear.pre_quant_scale is not None
                            else None
                        ),

                        "bias": (
                            torch.empty(
                                linear.bias.shape,
                                dtype=linear.bias.dtype,
                                device=self.device,
                            )
                            if linear.bias is not None
                            else None
                        ),
                    }
                )

            self.slots[
                slot_idx
            ] = buffers

        print(
            f"[Qwen3 H2D] GPU ring allocated: "
            f"{self.ring_size} layer slots",
            flush=True,
        )

    # ================================================================
    # Bind one physical slot to one decoder layer
    # ================================================================

    def _bind_slot(
        self,
        layer_idx,
        slot_idx,
    ):

        layer = self.layers[
            layer_idx
        ]

        linears = self._nvfp4_linears(
            layer
        )

        buffers = self.slots[
            slot_idx
        ]

        if len(linears) != len(buffers):

            raise RuntimeError(
                "NVFP4 linear count mismatch: "
                f"layer={len(linears)}, "
                f"slot={len(buffers)}"
            )

        for linear, b in zip(
            linears,
            buffers,
        ):

            linear._gpu_packed = (
                b["packed"]
            )

            linear._gpu_bscale = (
                b["bscale"]
            )

            linear._gpu_gscale = (
                b["gscale"]
            )

            linear._gpu_pqs = (
                b["pqs"]
            )

            linear._gpu_bias = (
                b["bias"]
            )

    # ================================================================
    # Move ONLY ordinary non-NVFP4 parameters of active layer to GPU
    # ================================================================

    def _layer_to_gpu(
        self,
        layer_idx,
    ):

        layer = self.layers[
            layer_idx
        ]

        # Nvfp4Linear._apply() intentionally keeps:
        #
        #   packed
        #   bscale
        #   gscale
        #   pre_quant_scale
        #   bias
        #
        # on CPU.
        #
        # Therefore layer.to(cuda) moves only the ordinary
        # parameters such as LayerNorm / attention parameters.

        layer.to(
            self.device,
            non_blocking=True,
        )

    # ================================================================
    # Move ordinary parameters of old layer back to CPU
    # ================================================================

    def _layer_to_cpu(
        self,
        layer_idx,
    ):

        if (
            layer_idx is None
            or layer_idx < 0
            or layer_idx >= self.n_layers
        ):
            return

        layer = self.layers[
            layer_idx
        ]

        layer.to(
            "cpu",
            non_blocking=True,
        )

    # ================================================================
    # Async H2D of one layer
    # ================================================================

    def _load(
        self,
        layer_idx,
        slot_idx,
    ):

        if layer_idx < 0:
            return

        if layer_idx >= self.n_layers:
            return

        if (
            self.loaded_layer[
                slot_idx
            ]
            == layer_idx
        ):
            return

        gate = self.free_event[
            slot_idx
        ]

        layer = self.layers[
            layer_idx
        ]

        linears = self._nvfp4_linears(
            layer
        )

        buffers = self.slots[
            slot_idx
        ]

        # ------------------------------------------------------------
        # H2D stream
        # ------------------------------------------------------------

        with torch.cuda.stream(
            self.copy_stream
        ):

            # The ring slot cannot be overwritten until the previous
            # compute stream has finished using it.
            if gate is not None:

                self.copy_stream.wait_event(
                    gate
                )

            for linear, b in zip(
                linears,
                buffers,
            ):

                # --------------------------------------------------
                # All CPU sources are pinned.
                #
                # non_blocking=True therefore allows the H2D copies
                # to be scheduled asynchronously on copy_stream.
                # --------------------------------------------------

                b["packed"].copy_(
                    linear.packed,
                    non_blocking=True,
                )

                b["bscale"].copy_(
                    linear.bscale,
                    non_blocking=True,
                )

                b["gscale"].copy_(
                    linear.gscale,
                    non_blocking=True,
                )

                if (
                    linear.pre_quant_scale
                    is not None
                ):

                    b["pqs"].copy_(
                        linear.pre_quant_scale,
                        non_blocking=True,
                    )

                if (
                    linear.bias
                    is not None
                ):

                    b["bias"].copy_(
                        linear.bias,
                        non_blocking=True,
                    )

            done = (
                self.copy_stream.record_event()
            )

        self.loaded_layer[
            slot_idx
        ] = layer_idx

        self.copy_done[
            layer_idx
        ] = done

        # ------------------------------------------------------------
        # Bind GPU buffers immediately.
        #
        # Actual compute waits for copy_done.
        # ------------------------------------------------------------

        self._bind_slot(
            layer_idx,
            slot_idx,
        )

    # ================================================================
    # Initial two-layer preload
    # ================================================================

    def prepare(self):

        self.loaded_layer = [
            None
        ] * self.ring_size

        self.copy_done = [
            None
        ] * self.n_layers

        self.free_event = [
            None
        ] * self.ring_size

        initial = min(
            self.ring_size,
            self.n_layers,
        )

        for layer_idx in range(
            initial
        ):

            slot_idx = (
                layer_idx
                % self.ring_size
            )

            self._load(
                layer_idx,
                slot_idx,
            )

        # Wait until initial layers are physically copied.
        event = (
            self.copy_stream.record_event()
        )

        torch.cuda.current_stream().wait_event(
            event
        )

        # Ordinary parameters of the first two layers.
            
        print(
            f"[Qwen3 H2D] prepared: "
            f"{self.n_layers} layers, "
            f"GPU slots={self.ring_size}",
            flush=True,
        )

    # ================================================================
    # Begin layer
    # ================================================================

    def begin_layer(
        self,
        layer_idx,
    ):

        if (
            layer_idx < 0
            or layer_idx >= self.n_layers
        ):
            return

        slot_idx = (
            layer_idx
            % self.ring_size
        )

        # Make sure this layer is loaded.
        if (
            self.loaded_layer[
                slot_idx
            ]
            != layer_idx
        ):

            self._load(
                layer_idx,
                slot_idx,
            )

        # Wait for H2D completion.
        event = self.copy_done[
            layer_idx
        ]

        if event is not None:

            torch.cuda.current_stream().wait_event(
                event
            )

        # LayerNorm etc.
        self._layer_to_gpu(
            layer_idx
        )

    # ================================================================
    # End layer
    # ================================================================

    def end_layer(
        self,
        layer_idx,
    ):

        if (
            layer_idx < 0
            or layer_idx >= self.n_layers
        ):
            return

        slot_idx = (
            layer_idx
            % self.ring_size
        )

        # Current compute stream has finished using
        # this slot when this event is reached.
        self.free_event[
            slot_idx
        ] = (
            torch.cuda.current_stream()
            .record_event()
        )

        # ------------------------------------------------------------
        # Start loading layer N + 2 into the freed slot.
        # ------------------------------------------------------------

        next_layer = (
            layer_idx
            + self.ring_size
        )

        if (
            next_layer
            < self.n_layers
        ):

            next_slot = (
                next_layer
                % self.ring_size
            )

            self._load(
                next_layer,
                next_slot,
            )

        # ------------------------------------------------------------
        # The old layer no longer needs to remain on GPU.
        #
        # Quant buffers and bias are protected by Nvfp4Linear._apply()
        # and therefore remain on CPU.
        #
        # Only ordinary layer parameters are moved back.
        # ------------------------------------------------------------

        old_layer = layer_idx

    # ================================================================
    # Compatibility API
    # ================================================================

    def wait_for_layer(
        self,
        layer_idx,
    ):

        self.begin_layer(
            layer_idx
        )

    def submit_move(
        self,
        layer_idx,
    ):

        self.end_layer(
            layer_idx
        )

    # ================================================================
    # Cleanup
    # ================================================================

    def unload_all(self):

        self.copy_stream.synchronize()

        for idx in range(
            self.n_layers
        ):

            self._layer_to_cpu(
                idx
            )

        for layer in self.layers:

            for linear in (
                self._nvfp4_linears(
                    layer
                )
            ):

                linear._gpu_packed = None
                linear._gpu_bscale = None
                linear._gpu_gscale = None
                linear._gpu_pqs = None
                linear._gpu_bias = None

        self.loaded_layer = [
            None
        ] * self.ring_size

        self.copy_done = [
            None
        ] * self.n_layers

    def synchronize(self):

        self.copy_stream.synchronize()


# ============================================================================
# QWEN3-VL LAYERWISE FORWARD
# ============================================================================

@torch.no_grad()
def qwen3vl_layerwise_forward(
    model,
    hidden_states,
    attention_mask,
    position_ids,
    cache_position=None,
    past_key_values=None,
    visual_pos_masks=None,
    deepstack_visual_embeds=None,
    device="cuda",
    layer_streamer=None,
):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        create_causal_mask,
    )

    lm = model.language_model
    gpu = torch.device(device)

    # Qwen3-VL rotary expects:
    #   [3, batch, seq]
    #
    # create_causal_mask expects position_ids after selecting
    # the text component:
    #   [batch, seq]

    if position_ids.ndim == 1:

        position_ids = position_ids.unsqueeze(0)

    if position_ids.ndim == 2:

        # [batch, seq] -> [3, batch, seq]
        position_ids = position_ids.unsqueeze(0).expand(
            3,
            -1,
            -1,
        )

    text_position_ids = position_ids[0]

    if cache_position is None:

        cache_position = torch.arange(
            hidden_states.shape[1],
            device=hidden_states.device,
        )

    attention_mask = create_causal_mask(
        config=lm.config,
        input_embeds=hidden_states,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )

    position_embeddings = lm.rotary_emb(
        hidden_states,
        position_ids,
    )

    for layer_idx, decoder_layer in enumerate(
        lm.layers
    ):

        if layer_streamer is not None:

            layer_streamer.begin_layer(
                layer_idx
            )

        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        if (
            deepstack_visual_embeds is not None
            and layer_idx < len(
                deepstack_visual_embeds
            )
        ):

            hidden_states = lm._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[
                    layer_idx
                ],
            )

        if layer_streamer is not None:

            layer_streamer.end_layer(
                layer_idx
            )

    hidden_states = lm.norm(
        hidden_states
    )

    return hidden_states


# ============================================================================
# TEXT ENCODER
# ============================================================================

class MiniMaxH3TextEncoder:

    def __init__(
        self,
        model,
        tokenizer,
        device="cuda",
        compute_dtype=torch.bfloat16,
        cpu_embed=False,
        layer_streaming=False,
    ):

        self.model = model.eval()

        self.tokenizer = tokenizer

        self.device = torch.device(
            device
        )

        self.compute_dtype = (
            compute_dtype
        )

        self.cpu_embed = bool(
            cpu_embed
        )

        self._cache = {}

        self._image_processor = None

        self.layer_streamer = None

        if layer_streaming:

            self.layer_streamer = (
                Qwen3VLLayerStreamer(
                    self.model,
                    self.device,
                    ring_size=2,
                )
            )

            self.layer_streamer.prepare()

    # ------------------------------------------------------------------
    # Decoder + embeddings
    # ------------------------------------------------------------------

    def _get_decoder_and_embeddings(
        self,
    ):

        if hasattr(
            self.model,
            "language_model",
        ):

            return (
                self.model.language_model,
                self.model.language_model.embed_tokens,
            )

        return (
            self.model,
            self.model.embed_tokens,
        )

    # ------------------------------------------------------------------
    # Embedding device
    # ------------------------------------------------------------------

    def _embedding_device(
        self,
    ):

        _, embed_tokens = (
            self._get_decoder_and_embeddings()
        )

        return embed_tokens.weight.device

    # ------------------------------------------------------------------
    # Text forward
    # ------------------------------------------------------------------

    def _text_forward(
        self,
        ids,
    ):

        decoder, embed_tokens = (
            self._get_decoder_and_embeddings()
        )

        # ==========================================================
        # STREAMING MODE
        #
        # НИКОГДА не вызываем self.model(...)
        # ==========================================================

        if self.layer_streamer is not None:

            if self.cpu_embed:

                ids_device = ids.to("cpu")

                emb = embed_tokens(
                    ids_device
                )

                ids_rope = ids.to(
                    self.device
                )

                emb = emb.to(
                    self.device,
                    self.compute_dtype,
                    non_blocking=True,
                )

            else:

                ids_rope = ids.to(
                    self.device
                )

                emb = embed_tokens(
                    ids_rope
                )

                emb = emb.to(
                    self.compute_dtype
                )

            position_ids, _ = (
                self.model.get_rope_index(
                    ids_rope,
                    None,
                    None,
                    attention_mask=None,
                )
            )

            position_ids = position_ids.to(
                emb.device
            )

            from transformers.cache_utils import DynamicCache

            # Not an optimization — a bitwise-parity requirement. The stock HF forward
            # creates a DynamicCache when past_key_values is None, and with a None
            # attention mask the cache's presence changes attention kernel selection:
            # without it the streamed text output drifts ~1e0 in bf16 (cosine 1.0,
            # token 0 exact — pure kernel noise). The reference path is immune (its
            # explicit ones-mask forces the same kernel either way).
            pkv = DynamicCache(
                config=self.model.language_model.config
            )

            return qwen3vl_layerwise_forward(
                self.model,
                emb,
                attention_mask=None,
                position_ids=position_ids,
                cache_position=None,
                past_key_values=pkv,
                device=self.device,
                layer_streamer=self.layer_streamer,
            )

        # ==========================================================
        # Обычный режим без streaming
        # ==========================================================

        if self.cpu_embed:

            emb = embed_tokens(
                ids.to("cpu")
            )

            emb = emb.to(
                self.device,
                self.compute_dtype,
                non_blocking=True,
            )

            return decoder(
                inputs_embeds=emb,
            ).last_hidden_state

        return self.model(
            input_ids=ids.to(
                self.device
            )
        ).last_hidden_state

    # ------------------------------------------------------------------
    # Reference forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _reference_forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
    ):

        model = self.model

        embedding_device = (
            self._embedding_device()
        )

        input_ids = input_ids.to(
            embedding_device,
            non_blocking=True,
        )

        if attention_mask is not None:

            if isinstance(
                attention_mask,
                dict,
            ):

                attention_mask = {
                    k: (
                        v.to(
                            embedding_device,
                            non_blocking=True,
                        )
                        if torch.is_tensor(v)
                        else v
                    )
                    for k, v in attention_mask.items()
                }

            else:

                attention_mask = (
                    attention_mask.to(
                        embedding_device,
                        non_blocking=True,
                    )
                )

        if pixel_values is not None:

            pixel_values = pixel_values.to(
                embedding_device,
                dtype=torch.bfloat16,
                non_blocking=True,
            )

        if image_grid_thw is not None:

            image_grid_thw = (
                image_grid_thw.to(
                    embedding_device,
                    non_blocking=True,
                )
            )

        inputs_embeds = (
            model.get_input_embeddings()(
                input_ids
            )
        )

        inputs_embeds = inputs_embeds.to(
            self.compute_dtype
        )

        image_mask = None
        deepstack_image_embeds = None

        if pixel_values is not None:

            (
                image_embeds,
                deepstack_image_embeds,
            ) = model.get_image_features(
                pixel_values,
                image_grid_thw,
            )

            image_embeds = torch.cat(
                image_embeds,
                dim=0,
            ).to(
                inputs_embeds.device,
                inputs_embeds.dtype,
            )

            image_mask, _ = (
                model.get_placeholder_mask(
                    input_ids,
                    inputs_embeds=inputs_embeds,
                    image_features=image_embeds,
                )
            )

            inputs_embeds = (
                inputs_embeds.masked_scatter(
                    image_mask,
                    image_embeds,
                )
            )

        visual_pos_masks = None
        deepstack_visual_embeds = None

        if image_mask is not None:

            image_mask = image_mask[..., 0]

            visual_pos_masks = image_mask

            deepstack_visual_embeds = (
                deepstack_image_embeds
            )

            if deepstack_visual_embeds is not None:

                deepstack_visual_embeds = [
                    x.to(
                        inputs_embeds.device,
                        inputs_embeds.dtype,
                    )
                    for x in deepstack_visual_embeds
                ]

        attention_mask_tensor = (
            attention_mask
        )

        if isinstance(
            attention_mask_tensor,
            dict,
        ):

            attention_mask_tensor = (
                attention_mask_tensor[
                    "full_attention"
                ]
            )

        if (
            attention_mask_tensor is not None
            and attention_mask_tensor.ndim == 4
        ):

            attention_mask_tensor = (
                torch.diagonal(
                    attention_mask_tensor[:, 0],
                    dim1=1,
                    dim2=2,
                )
            )

            if (
                attention_mask_tensor.dtype
                .is_floating_point
            ):

                attention_mask_tensor = (
                    attention_mask_tensor
                    / torch.finfo(
                        attention_mask_tensor.dtype
                    ).min
                )

                attention_mask_tensor = (
                    1.0
                    - attention_mask_tensor
                ).int()

        if attention_mask_tensor is not None:

            attention_mask_tensor = (
                attention_mask_tensor.to(
                    embedding_device
                )
            )

        position_ids, rope_deltas = (
            model.get_rope_index(
                input_ids,
                image_grid_thw,
                None,
                attention_mask=attention_mask_tensor,
            )
        )

        position_ids = position_ids.to(
            embedding_device
        )

        if rope_deltas is not None:

            rope_deltas = rope_deltas.to(
                embedding_device
            )

        model.rope_deltas = rope_deltas

        if self.layer_streamer is not None:

            outputs = qwen3vl_layerwise_forward(
                model,
                inputs_embeds,
                attention_mask,
                position_ids,
                past_key_values=None,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                device=self.device,
                layer_streamer=self.layer_streamer,
            )

        else:

            outputs = model.language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                cache_position=None,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
            ).last_hidden_state

        return outputs

    # ------------------------------------------------------------------
    # PAD
    # ------------------------------------------------------------------

    def _pad_id(
        self,
    ):

        pid = getattr(
            self.tokenizer,
            "pad_token_id",
            None,
        )

        return (
            151643
            if pid is None
            else int(pid)
        )

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        caption: str,
        max_length: int = None,
    ):

        hit = self._cache.get(
            caption
        )

        if hit is not None:

            return hit.clone()

        _tk = dict(
            add_special_tokens=False,
            return_tensors="pt",
        )

        if max_length:

            _tk.update(
                truncation=True,
                max_length=max_length,
            )

        ids = self.tokenizer(
            caption,
            **_tk,
        )[
            "input_ids"
        ]

        if ids.shape[1] == 0:

            ids = torch.tensor(
                [
                    [
                        self._pad_id()
                    ]
                ],
                dtype=torch.long,
            )

        emb = self._text_forward(
            ids
        ).to(
            self.compute_dtype
        )

        self._cache[
            caption
        ] = emb.detach().to(
            "cpu"
        )

        return emb

    # ------------------------------------------------------------------
    # Reference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_with_reference(
        self,
        caption: str,
        images,
        max_length: int = None,
    ):

        if not hasattr(
            self.model,
            "visual",
        ):

            raise RuntimeError(
                "encode_with_reference needs "
                "the vision-capable encoder."
            )

        if self._image_processor is None:

            self._image_processor = (
                build_image_processor()
            )

        (
            ids,
            tags,
            pixel_values,
            grid,
        ) = build_reference_tokens(
            self.tokenizer,
            self._image_processor,
            caption,
            images,
            max_length,
        )

        attention_mask = torch.ones_like(
            ids
        )

        emb = self._reference_forward(
            input_ids=ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid,
        )

        emb = emb.to(
            self.compute_dtype
        )

        return emb, tags

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_batch(
        self,
        captions,
        max_length: int = None,
        batch_size: int = 8,
    ):

        out = [
            None
        ] * len(captions)

        todo = [
            i
            for i, c in enumerate(captions)
            if c not in self._cache
        ]

        for i, c in enumerate(
            captions
        ):

            if c in self._cache:

                out[i] = (
                    self._cache[c].clone()
                )

        pad_id = self._pad_id()

        for start in range(
            0,
            len(todo),
            batch_size,
        ):

            idxs = todo[
                start:start + batch_size
            ]

            toks = []

            for i in idxs:

                _tk = dict(
                    add_special_tokens=False,
                    return_tensors="pt",
                )

                if max_length:

                    _tk.update(
                        truncation=True,
                        max_length=max_length,
                    )

                t = self.tokenizer(
                    captions[i],
                    **_tk,
                )[
                    "input_ids"
                ][0]

                if t.numel():

                    toks.append(t)

                else:

                    toks.append(
                        torch.tensor(
                            [pad_id],
                            dtype=torch.long,
                        )
                    )

            L = max(
                t.numel()
                for t in toks
            )

            ids = torch.full(
                (
                    len(toks),
                    L,
                ),
                pad_id,
                dtype=torch.long,
            )

            for r, t in enumerate(
                toks
            ):

                ids[
                    r,
                    :t.numel()
                ] = t

            # ------------------------------------------------------
            # В streaming режиме здесь будет:
            #
            # embed -> H2D -> layer 0
            #                     + H2D layer 2
            #             -> layer 1
            #                     + H2D layer 3
            #             ...
            #
            # Никакого обычного model.forward().
            # ------------------------------------------------------

            hs = self._text_forward(
                ids
            )

            for r, i in enumerate(
                idxs
            ):

                emb = (
                    hs[
                        r,
                        :toks[r].numel()
                    ]
                    .unsqueeze(0)
                    .to(
                        self.compute_dtype
                    )
                )

                self._cache[
                    captions[i]
                ] = (
                    emb.detach()
                    .to("cpu")
                )

                out[i] = emb

        return out


# ============================================================================
# LOADER
# ============================================================================

def load_minimax_h3_te(
    path: str,
    device="cuda",
    compute_dtype=torch.bfloat16,
    quantize=True,
    tokenizer_dir=None,
    te_quant="auto",
    with_vision=False,
    cpu_embed=True,
    layer_streaming=True,
) -> MiniMaxH3TextEncoder:

    from bitsandbytes.nn import (
        Linear4bit,
        Params4bit,
    )

    from transformers import (
        AutoTokenizer,
    )

    cpu_embed = bool(
        cpu_embed
    )

    # ------------------------------------------------------------------
    # Detect checkpoint
    # ------------------------------------------------------------------

    with MemoryEfficientSafeOpen(
        path
    ) as _probe:

        _is_cq = any(
            k.endswith(
                ".comfy_quant"
            )
            for k in _probe.keys()
        )

    mode = te_quant

    if mode == "auto":

        mode = (
            "nvfp4"
            if _is_cq
            else "nf4"
        )

    if (
        mode == "nvfp4"
        and not _is_cq
    ):

        raise ValueError(
            "te_quant='nvfp4' needs "
            "the nvfp4-awq checkpoint"
        )

    if not quantize:

        mode = "none"

    if with_vision:

        cpu_embed = False

    streaming_enabled = (
        layer_streaming
        and mode == "nvfp4"
        and with_vision
    )

    # ------------------------------------------------------------------
    # Construct model on meta
    # ------------------------------------------------------------------

    with torch.device("meta"):

        if with_vision:

            model = build_qwen3vl_te()

        else:

            model = build_qwen3_te()

        if mode != "none":

            for (
                mod_name,
                module,
            ) in list(
                model.named_modules()
            ):

                for (
                    child_name,
                    child,
                ) in list(
                    module.named_children()
                ):

                    full = (
                        f"{mod_name}.{child_name}"
                        if mod_name
                        else child_name
                    )

                    if not (
                        isinstance(
                            child,
                            nn.Linear,
                        )
                        and (
                            full
                            + ".weight"
                        ).endswith(
                            _NF4_SUFFIXES
                        )
                    ):

                        continue

                    if mode == "nvfp4":

                        setattr(
                            module,
                            child_name,
                            Nvfp4Linear(
                                child.in_features,
                                child.out_features,
                                bias=(
                                    child.bias
                                    is not None
                                ),
                                compute_dtype=(
                                    compute_dtype
                                ),
                            ),
                        )

                    else:

                        q = Linear4bit(
                            child.in_features,
                            child.out_features,
                            bias=(
                                child.bias
                                is not None
                            ),
                            compute_dtype=(
                                compute_dtype
                            ),
                            quant_type="nf4",
                        )

                        setattr(
                            module,
                            child_name,
                            q,
                        )

    dev = torch.device(
        device
    )

    print(
        "[load] streaming the "
        "Qwen3-VL-32B text encoder...",
        flush=True,
    )

    model_keys = {
        n
        for n, _
        in model.named_parameters()
    }

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------

    with MemoryEfficientSafeOpen(
        path
    ) as f:

        ckpt = set(
            f.keys()
        )

        is_comfy_quant = any(
            k.endswith(
                ".comfy_quant"
            )
            for k in ckpt
        )

        if mode == "nvfp4":

            print(
                "[minimax-te] keeping packed "
                "NVFP4 weights on CPU; "
                "2-layer H2D streaming enabled",
                flush=True,
            )

        def _ck(
            name: str,
        ) -> str:

            if not with_vision:

                return (
                    "model."
                    + name
                )

            if name.startswith(
                "language_model."
            ):

                return (
                    "model."
                    + name[
                        len(
                            "language_model."
                        ):
                    ]
                )

            return name

        # --------------------------------------------------------------
        # NVFP4 packed tensors
        #
        # IMPORTANT:
        # All CPU sources are pinned so copy_(non_blocking=True)
        # can actually perform asynchronous H2D.
        # --------------------------------------------------------------

        if mode == "nvfp4":

            for (
                mod_name,
                module,
            ) in model.named_modules():

                if not isinstance(
                    module,
                    Nvfp4Linear,
                ):

                    continue

                fm = _ck(
                    mod_name
                )

                module.packed = (
                    f.get_tensor(
                        fm + ".weight"
                    )
                    .to("cpu")
                    .contiguous()
                    .pin_memory()
                )

                module.bscale = (
                    f.get_tensor(
                        fm + ".weight_scale"
                    )
                    .to("cpu")
                    .contiguous()
                    .view(
                        torch.uint8
                    )
                    .pin_memory()
                )

                module.gscale = (
                    f.get_tensor(
                        fm + ".weight_scale_2"
                    )
                    .to(
                        torch.float32
                    )
                    .reshape(1)
                    .contiguous()
                    .view(
                        torch.uint8
                    )
                    .to("cpu")
                    .pin_memory()
                )

                pqs_key = (
                    fm
                    + ".pre_quant_scale"
                )

                if pqs_key in ckpt:

                    module.pre_quant_scale = (
                        f.get_tensor(
                            pqs_key
                        )
                        .to(
                            compute_dtype
                        )
                        .to("cpu")
                        .contiguous()
                        .pin_memory()
                    )

                else:

                    module.pre_quant_scale = None

        # --------------------------------------------------------------
        # Other parameters
        # --------------------------------------------------------------

        for name in model_keys:

            src = _ck(
                name
            )

            file_mod = (
                src.rsplit(
                    ".",
                    1,
                )[0]
            )

            leaf = (
                name.rsplit(
                    ".",
                    1,
                )[1]
            )

            # ----------------------------------------------------------
            # Comfy quant
            # ----------------------------------------------------------

            if (
                is_comfy_quant
                and leaf == "weight"
                and (
                    file_mod
                    + ".comfy_quant"
                ) in ckpt
            ):

                if (
                    mode == "nvfp4"
                    and name.endswith(
                        _NF4_SUFFIXES
                    )
                ):

                    continue

                w = _dequant_comfy_weight(
                    f,
                    file_mod,
                    ckpt,
                )

            elif src in ckpt:

                w = f.get_tensor(
                    src
                )

            else:

                continue

            parent = model.get_submodule(
                name.rsplit(
                    ".",
                    1,
                )[0]
            )

            # ----------------------------------------------------------
            # NVFP4 linears:
            #
            # packed data was loaded above.
            # ----------------------------------------------------------

            if (
                mode == "nvfp4"
                and name.endswith(
                    _NF4_SUFFIXES
                )
            ):

                continue

            # ----------------------------------------------------------
            # NF4
            # ----------------------------------------------------------

            if (
                mode == "nf4"
                and name.endswith(
                    _NF4_SUFFIXES
                )
            ):

                p = Params4bit(
                    w.to(
                        compute_dtype
                    ),
                    requires_grad=False,
                    quant_type="nf4",
                )

                setattr(
                    parent,
                    leaf,
                    p,
                )

            else:

                keep = (
                    w.to(torch.float32)
                    if w.dtype
                    == torch.float32
                    else w.to(
                        compute_dtype
                    )
                )

                # ------------------------------------------------------
                # Embedding
                # ------------------------------------------------------

                if (
                    name
                    == "embed_tokens.weight"
                    and cpu_embed
                ):

                    tgt = torch.device(
                        "cpu"
                    )

                elif (
                    with_vision
                    and name
                    == "language_model.embed_tokens.weight"
                ):

                    tgt = dev

                elif (
                    with_vision
                    and not name.startswith(
                        "language_model."
                    )
                ):

                    tgt = dev

                else:

                    if streaming_enabled:

                        tgt = torch.device(
                            "cpu"
                        )

                    else:

                        tgt = dev

                param = nn.Parameter(
                    keep.to(tgt),
                    requires_grad=False,
                )

                # ------------------------------------------------------
                # Bias is part of the ring exchange for NVFP4.
                #
                # Keep it pinned on CPU so the copy stream can perform
                # asynchronous H2D.
                # ------------------------------------------------------

                if (
                    streaming_enabled
                    and name.endswith(".bias")
                    and param.device.type == "cpu"
                ):

                    param = nn.Parameter(
                        param.detach().pin_memory(),
                        requires_grad=False,
                    )

                setattr(
                    parent,
                    leaf,
                    param,
                )

    # ------------------------------------------------------------------
    # Rotary embeddings
    # ------------------------------------------------------------------

    if with_vision:

        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextRotaryEmbedding,
            Qwen3VLVisionRotaryEmbedding,
        )

        model.language_model.rotary_emb = (
            Qwen3VLTextRotaryEmbedding(
                model.config.text_config
            ).to(dev)
        )

        _vcfg = (
            model.config.vision_config
        )

        model.visual.rotary_pos_emb = (
            Qwen3VLVisionRotaryEmbedding(
                _vcfg.hidden_size
                // _vcfg.num_heads
                // 2
            ).to(dev)
        )

    else:

        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3RotaryEmbedding,
        )

        model.rotary_emb = (
            Qwen3RotaryEmbedding(
                model.config
            ).to(dev)
        )

    # ------------------------------------------------------------------
    # Meta buffers
    # ------------------------------------------------------------------

    for mod in model.modules():

        for (
            bname,
            buf,
        ) in list(
            mod.named_buffers(
                recurse=False
            )
        ):

            if (
                buf is not None
                and buf.is_meta
            ):

                mod.register_buffer(
                    bname,
                    torch.zeros(
                        buf.shape,
                        dtype=buf.dtype,
                        device=dev,
                    ),
                )

    # ------------------------------------------------------------------
    # Freeze
    # ------------------------------------------------------------------

    model.requires_grad_(False)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    tok = AutoTokenizer.from_pretrained(
        tokenizer_dir
        or _bundled_tokenizer_dir()
    )

    _add_h3_special_tokens(
        tok
    )

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    encoder = MiniMaxH3TextEncoder(
        model,
        tok,
        device=device,
        compute_dtype=compute_dtype,
        cpu_embed=cpu_embed,
        layer_streaming=streaming_enabled,
    )

    return encoder