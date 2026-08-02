"""Core training loop for Klein 9B LoRA training.

KleinTrainer handles Klein 9B image LoRA training end to end: optimizer /
scheduler setup, the denoising loss step, network save/load, adaptive-LR bookkeeping,
sample generation, and the resumable training-state snapshot.
"""

import ast
import argparse
import gc
import importlib
import json
import math
import os
import pathlib
import random
import re
import sys
import time
from datetime import timedelta
from multiprocessing import Value
from typing import Any, Dict, List, Optional

import accelerate
import numpy as np
import toml
import torch
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs, PartialState
from accelerate.utils import DynamoBackend, TorchDynamoPlugin, set_seed
from diffusers.optimization import (
    SchedulerType as DiffusersSchedulerType,
    TYPE_TO_SCHEDULER_FUNCTION as DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION,
)
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from packaging.version import Version
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm
from transformers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION

from fizgig.dataset.config import BlueprintGenerator, ConfigSanitizer, generate_dataset_group_by_blueprint, load_user_config
from fizgig.klein.model import KleinDiT, Klein9BParams
from fizgig.klein.model_utils import (
    KLEIN_MODEL_INFO,
    KleinModelInfo,
    load_dit,
    load_vae,
    load_text_encoder,
    get_schedule,
    denoise,
    denoise_cfg,
)
from fizgig.klein.position import prc_img, prc_txt, scatter_ids, pack_control_latent
from fizgig.modules.schedulers import FlowMatchDiscreteScheduler, RexLR
from fizgig.training.metadata import (
    build_metadata, latest_sample_image, thumbnail_data_uri, resolve_title,
    ARCHITECTURE_KLEIN_9B, ARCHITECTURE_KLEIN_9B_FULL,
)
from fizgig.training.train_utils import (
    LossRecorder,
    get_epoch_ckpt_name,
    get_step_ckpt_name,
    get_last_ckpt_name,
    get_remove_epoch_no,
    get_remove_step_no,
    get_sanitized_config_or_none,
    get_lin_function,
    prune_state_dirs,
    save_state_on_epoch_end,
    save_and_remove_state_stepwise,
    save_state_on_train_end,
)
from fizgig.utils.device import clean_memory_on_device
import fizgig.networks.lora as lora_module

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Metadata keys
# ---------------------------------------------------------------------------

SS_METADATA_KEY_BASE_MODEL_VERSION = "ss_base_model_version"
SS_METADATA_KEY_NETWORK_MODULE = "ss_network_module"
SS_METADATA_KEY_NETWORK_DIM = "ss_network_dim"
SS_METADATA_KEY_NETWORK_ALPHA = "ss_network_alpha"
SS_METADATA_KEY_NETWORK_ARGS = "ss_network_args"

SS_METADATA_MINIMUM_KEYS = [
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
]


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class _Collator:
    """Collator for the dataloader — batch size is always 1, so we just unwrap."""

    def __init__(self, epoch, dataset):
        self.current_epoch = epoch
        self.dataset = dataset

    def __call__(self, examples):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset if worker_info is not None else self.dataset
        dataset.set_current_epoch(self.current_epoch.value)
        return examples[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prepare_accelerator(args: argparse.Namespace) -> Accelerator:
    """Build an Accelerator from parsed args."""
    if args.logging_dir is None:
        logging_dir = None
    else:
        log_prefix = "" if args.log_prefix is None else args.log_prefix
        logging_dir = args.logging_dir + "/" + log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime())

    if args.log_with is None:
        log_with = "tensorboard" if logging_dir is not None else None
    else:
        log_with = args.log_with
        if log_with in ["tensorboard", "all"]:
            if logging_dir is None:
                raise ValueError("logging_dir is required when log_with is tensorboard")
        if log_with in ["wandb", "all"]:
            try:
                import wandb
            except ImportError:
                raise ImportError("wandb is not installed")
            if logging_dir is not None:
                os.makedirs(logging_dir, exist_ok=True)
                os.environ["WANDB_DIR"] = logging_dir
            if args.wandb_api_key is not None:
                wandb.login(key=args.wandb_api_key)

    kwargs_handlers = [
        (
            InitProcessGroupKwargs(
                backend="gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
                init_method=(
                    "env://?use_libuv=False"
                    if os.name == "nt" and Version(torch.__version__) >= Version("2.4.0")
                    else None
                ),
                timeout=timedelta(minutes=args.ddp_timeout) if args.ddp_timeout else None,
            )
            if torch.cuda.device_count() > 1
            else None
        ),
        (
            DistributedDataParallelKwargs(
                gradient_as_bucket_view=args.ddp_gradient_as_bucket_view,
                static_graph=args.ddp_static_graph,
            )
            if args.ddp_gradient_as_bucket_view or args.ddp_static_graph
            else None
        ),
    ]
    kwargs_handlers = [h for h in kwargs_handlers if h is not None]

    dynamo_plugin = None
    if args.dynamo_backend.upper() != "NO":
        dynamo_plugin = TorchDynamoPlugin(
            backend=DynamoBackend(args.dynamo_backend.upper()),
            mode=args.dynamo_mode,
            fullgraph=args.dynamo_fullgraph,
            dynamic=args.dynamo_dynamic,
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if args.mixed_precision else None,
        log_with=log_with,
        project_dir=logging_dir,
        dynamo_plugin=dynamo_plugin,
        kwargs_handlers=kwargs_handlers,
    )
    print("accelerator device:", accelerator.device)
    return accelerator


def line_to_prompt_dict(line: str) -> dict:
    """Parse a prompt line with optional --args into a dict."""
    prompt_args = line.split(" --")
    prompt_dict = {"prompt": prompt_args[0]}

    for parg in prompt_args[1:]:
        try:
            m = re.match(r"w (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["width"] = int(m.group(1))
                continue
            m = re.match(r"h (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["height"] = int(m.group(1))
                continue
            m = re.match(r"d (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["seed"] = int(m.group(1))
                continue
            m = re.match(r"s (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["sample_steps"] = max(1, min(1000, int(m.group(1))))
                continue
            m = re.match(r"g ([\d\.]+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["guidance_scale"] = float(m.group(1))
                continue
            m = re.match(r"fs ([\d\.]+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["discrete_flow_shift"] = float(m.group(1))
                continue
            m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["cfg_scale"] = float(m.group(1))
                continue
            m = re.match(r"n (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["negative_prompt"] = m.group(1)
                continue
            m = re.match(r"ci (.+)", parg, re.IGNORECASE)
            if m:
                control_image_path = m.group(1).strip()
                if "control_image_path" not in prompt_dict:
                    prompt_dict["control_image_path"] = []
                prompt_dict["control_image_path"].append(control_image_path)
                continue
        except ValueError as ex:
            logger.error(f"Error parsing prompt arg: {parg}")
            logger.error(ex)

    return prompt_dict


def load_prompts(prompt_file: str) -> list[Dict]:
    """Read prompt file (.txt, .toml, or .json) into a list of prompt dicts."""
    if prompt_file.endswith(".txt"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    elif prompt_file.endswith(".toml"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = toml.load(f)
        prompts = [dict(**data["prompt"], **subset) for subset in data["prompt"]["subset"]]
    elif prompt_file.endswith(".json"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)
    else:
        raise ValueError(f"Unsupported prompt file format: {prompt_file}")

    for i in range(len(prompts)):
        prompt_dict = prompts[i]
        if isinstance(prompt_dict, str):
            prompt_dict = line_to_prompt_dict(prompt_dict)
            prompts[i] = prompt_dict
        assert isinstance(prompt_dict, dict)
        prompt_dict["enum"] = i
        prompt_dict.pop("subset", None)

    return prompts


def read_config_from_file(args: argparse.Namespace, parser: argparse.ArgumentParser):
    """If --config_file is specified, load TOML and merge into args."""
    if not args.config_file:
        return args

    config_path = args.config_file + ".toml" if not args.config_file.endswith(".toml") else args.config_file
    if not os.path.exists(config_path):
        logger.error(f"{config_path} not found.")
        sys.exit(1)

    logger.info(f"Loading settings from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = toml.load(f)

    ignore_nesting_dict = {}
    for section_name, section_dict in config_dict.items():
        if not isinstance(section_dict, dict):
            ignore_nesting_dict[section_name] = section_dict
            continue
        for key, value in section_dict.items():
            ignore_nesting_dict[key] = value

    config_args = argparse.Namespace(**ignore_nesting_dict)
    args = parser.parse_args(namespace=config_args)
    args.config_file = os.path.splitext(args.config_file)[0]
    logger.info(args.config_file)
    return args


# ---------------------------------------------------------------------------
# Timestep sampling helpers
# ---------------------------------------------------------------------------

def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """Compute density for SD3-style timestep sampling."""
    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


def get_sigmas(noise_scheduler, timesteps, device, n_dim=4, dtype=torch.float32):
    """Get sigma values from noise scheduler for given timesteps."""
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)

    if any([(schedule_timesteps == t).sum() == 0 for t in timesteps]):
        logger.warning("Some timesteps are not in the schedule — rounding to nearest")
        step_indices = [torch.argmin(torch.abs(schedule_timesteps - t)).item() for t in timesteps]
    else:
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def compute_loss_weighting_for_sd3(weighting_scheme: str, noise_scheduler, timesteps, device, dtype):
    """Compute loss weighting for SD3-style training."""
    if weighting_scheme == "sigma_sqrt" or weighting_scheme == "cosmap":
        sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=5, dtype=dtype)
        if weighting_scheme == "sigma_sqrt":
            weighting = (sigmas**-2.0).float()
        else:
            bot = 1 - 2 * sigmas + 2 * sigmas**2
            weighting = 2 / (math.pi * bot)
    else:
        weighting = None
    return weighting


def should_sample_images(args, steps, epoch=None):
    """Determine whether to generate sample images at this step/epoch."""
    if steps == 0:
        return args.sample_at_first
    should_by_steps = args.sample_every_n_steps is not None and steps % args.sample_every_n_steps == 0
    should_by_epochs = (
        args.sample_every_n_epochs is not None and epoch is not None and epoch % args.sample_every_n_epochs == 0
    )
    return should_by_steps or should_by_epochs


# ---------------------------------------------------------------------------
# KleinTrainer
# ---------------------------------------------------------------------------

class KleinTrainer:
    """Single-class LoRA trainer for Klein 9B image generation models."""

    def __init__(self):
        self.blocks_to_swap = None
        self.timestep_range_pool = []
        self.num_timestep_buckets: Optional[int] = None
        self.default_guidance_scale = 4.0  # CFG scale for inference (base models)
        self.default_discrete_flow_shift = None  # dynamic empirical mu
        self.model_version_info: Optional[KleinModelInfo] = None
        self.dit_dtype = torch.bfloat16
        self.context_network = None

    # ------------------------------------------------------------------
    # Architecture properties (hardcoded to Klein 9B)
    # ------------------------------------------------------------------

    @property
    def architecture(self) -> str:
        return self.model_version_info.architecture if self.model_version_info else ARCHITECTURE_KLEIN_9B

    @property
    def architecture_full_name(self) -> str:
        return self.model_version_info.architecture_full if self.model_version_info else ARCHITECTURE_KLEIN_9B_FULL

    # ------------------------------------------------------------------
    # Model-specific setup
    # ------------------------------------------------------------------

    def handle_model_specific_args(self, args: argparse.Namespace):
        """Initialize Klein model info from parsed args."""
        self.model_version_info = KLEIN_MODEL_INFO[args.model_version]
        self.dit_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
        self.default_guidance_scale = float(self.model_version_info.defaults.get("guidance", 3.5))

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        """Load the KleinDiT model."""
        model = load_dit(
            device=accelerator.device,
            model_version_info=self.model_version_info,
            dit_path=dit_path,
            attn_mode=attn_mode,
            split_attn=split_attn,
            loading_device=loading_device,
            dit_weight_dtype=dit_weight_dtype,
            fp8_scaled=args.fp8_scaled,
            disable_numpy_memmap=args.disable_numpy_memmap,
            quant_4bit=getattr(args, "quant_4bit", False),
        )
        return model

    def load_vae_model(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        """Load the AutoEncoder (VAE)."""
        logger.info(f"Loading AE model from {vae_path}")
        ae = load_vae(vae_path, dtype=vae_dtype, device="cpu", disable_mmap=True)
        return ae

    def compile_transformer(self, args, transformer):
        """Optional torch.compile of transformer blocks."""
        from fizgig.klein.model import KleinDiT as _KleinDiT

        model: _KleinDiT = transformer

        target_blocks = [model.double_blocks, model.single_blocks]
        disable_linear = self.blocks_to_swap > 0

        if disable_linear:
            logger.info("Disabling linear from torch.compile for swap blocks...")
            for blocks in target_blocks:
                for block in blocks:
                    for sub_module in block.modules():
                        if sub_module.__class__.__name__.endswith("Linear"):
                            if not hasattr(sub_module, "_forward_before_disable_compile"):
                                sub_module._forward_before_disable_compile = sub_module.forward
                                sub_module._eager_forward = torch._dynamo.disable()(sub_module.forward)
                            sub_module.forward = sub_module._eager_forward

        compile_dynamic = None
        if args.compile_dynamic is not None:
            compile_dynamic = {"true": True, "false": False, "auto": None}[args.compile_dynamic.lower()]

        logger.info(
            f"Compiling DiT with torch.compile: backend={args.compile_backend}, "
            f"mode={args.compile_mode}, dynamic={compile_dynamic}, fullgraph={args.compile_fullgraph}"
        )

        # Same two pre-compile fixes Krea 2 has: raise the recompile ceilings + sympy Mod
        # patch (default cache limit of 8 meant a bucketed dataset silently dropped to
        # permanent eager), and settle the SDPA backend global so its lazy first-use probe
        # never executes inside a compiled block.
        from fizgig.modules.compile_util import init_compile
        init_compile(args.compile_cache_size_limit if args.compile_cache_size_limit is not None else 8192)
        from fizgig.modules import sdpa as _sdpa_mod
        _sdpa_mod.prime()

        for blocks in target_blocks:
            for i, block in enumerate(blocks):
                block = torch.compile(
                    block,
                    backend=args.compile_backend,
                    mode=args.compile_mode,
                    dynamic=compile_dynamic,
                    fullgraph=args.compile_fullgraph,
                )
                blocks[i] = block

        return transformer

    # ------------------------------------------------------------------
    # Log generation
    # ------------------------------------------------------------------

    def generate_step_logs(
        self,
        args: argparse.Namespace,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions,
        optimizer=None,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
    ):
        logs = {"loss/current": current_loss, "loss/average": avr_loss}

        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/average_key_norm"] = mean_norm
            logs["max_norm/max_key_norm"] = maximum_norm

        lrs = lr_scheduler.get_last_lr()
        for i, lr in enumerate(lrs):
            if lr_descriptions is not None:
                lr_desc = lr_descriptions[i]
            else:
                idx = i
                if len(lrs) > 2:
                    lr_desc = f"group{i}"
                else:
                    lr_desc = "unet"
            logs[f"lr/{lr_desc}"] = lr

            if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower().endswith(
                "Prodigy".lower()
            ):
                logs[f"lr/d*lr/{lr_desc}"] = (
                    lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                )

            if args.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower()) and optimizer is not None:
                logs[f"lr/d*lr/{lr_desc}"] = optimizer.param_groups[i]["d"] * optimizer.param_groups[i]["lr"]
                if "effective_lr" in optimizer.param_groups[i]:
                    logs[f"lr/d*eff_lr/{lr_desc}"] = (
                        optimizer.param_groups[i]["d"] * optimizer.param_groups[i]["effective_lr"]
                    )

        return logs

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def get_optimizer(self, args, trainable_params: list[torch.nn.Parameter]) -> tuple:
        """Create optimizer from args. Returns (name, args_str, optimizer, train_fn, eval_fn)."""
        optimizer_type = args.optimizer_type.lower()

        optimizer_kwargs = {}
        if args.optimizer_args is not None and len(args.optimizer_args) > 0:
            for arg in args.optimizer_args:
                if "=" not in arg:
                    raise ValueError(f"--optimizer_args entry {arg!r} is not key=value")
                # maxsplit=1 (values may contain '='); non-literals like name=cosine or
                # growth_rate=inf stay strings instead of killing the run after caching.
                key, value = arg.split("=", 1)
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
                optimizer_kwargs[key] = value

        lr = args.learning_rate
        optimizer = None
        optimizer_class = None

        if optimizer_type.endswith("8bit"):
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError("bitsandbytes is not installed")

            if optimizer_type == "adamw8bit":
                logger.info(f"Using 8-bit AdamW optimizer | {optimizer_kwargs}")
                optimizer_class = bnb.optim.AdamW8bit
                optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "adamw":
            logger.info(f"Using AdamW optimizer | {optimizer_kwargs}")
            optimizer_class = torch.optim.AdamW
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        if optimizer is None:
            # Arbitrary optimizer from module path
            case_sensitive_optimizer_type = args.optimizer_type
            logger.info(f"Using {case_sensitive_optimizer_type} | {optimizer_kwargs}")

            if "." not in case_sensitive_optimizer_type:
                optimizer_module = torch.optim
            else:
                values = case_sensitive_optimizer_type.split(".")
                optimizer_module = importlib.import_module(".".join(values[:-1]))
                case_sensitive_optimizer_type = values[-1]

            try:
                optimizer_class = getattr(optimizer_module, case_sensitive_optimizer_type)
            except AttributeError:
                raise ValueError(
                    f"Unknown optimizer {args.optimizer_type!r}. Use adamw, adamw8bit, or a "
                    f"full module.path.ClassName (e.g. bitsandbytes.optim.AdEMAMix8bit). "
                    f"(adafactor/prodigy/came were removed — they manage their own LR and "
                    f"conflict with Adaptive LR.)"
                ) from None
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        optimizer_name = optimizer_class.__module__ + "." + optimizer_class.__name__
        optimizer_args = ",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])

        if hasattr(optimizer, "train") and callable(optimizer.train):
            train_fn = optimizer.train
            eval_fn = optimizer.eval
        else:
            train_fn = lambda: None
            eval_fn = lambda: None

        return optimizer_name, optimizer_args, optimizer, train_fn, eval_fn

    # ------------------------------------------------------------------
    # LR Scheduler
    # ------------------------------------------------------------------

    def is_schedulefree_optimizer(self, optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> bool:
        return args.optimizer_type.lower().endswith("schedulefree")

    def get_dummy_scheduler(self, optimizer: torch.optim.Optimizer) -> Any:
        """Dummy scheduler for schedule-free optimizers (logging only)."""

        class DummyScheduler:
            def __init__(self, opt):
                self.optimizer = opt

            def step(self):
                pass

            def get_last_lr(self):
                return [group["lr"] for group in self.optimizer.param_groups]

        return DummyScheduler(optimizer)

    def get_lr_scheduler(self, args, optimizer: torch.optim.Optimizer, num_processes: int):
        """Build the LR scheduler from args."""
        if self.is_schedulefree_optimizer(optimizer, args):
            return self.get_dummy_scheduler(optimizer)

        name = args.lr_scheduler
        num_training_steps = args.max_train_steps * num_processes
        num_warmup_steps: Optional[int] = (
            int(args.lr_warmup_steps * num_training_steps) if isinstance(args.lr_warmup_steps, float) else args.lr_warmup_steps
        )
        num_decay_steps: Optional[int] = (
            int(args.lr_decay_steps * num_training_steps) if isinstance(args.lr_decay_steps, float) else args.lr_decay_steps
        )
        num_stable_steps = num_training_steps - num_warmup_steps - num_decay_steps
        num_cycles = args.lr_scheduler_num_cycles
        power = args.lr_scheduler_power
        timescale = args.lr_scheduler_timescale
        min_lr_ratio = args.lr_scheduler_min_lr_ratio

        lr_scheduler_kwargs = {}
        if args.lr_scheduler_args is not None and len(args.lr_scheduler_args) > 0:
            for arg in args.lr_scheduler_args:
                key, value = arg.split("=")
                value = ast.literal_eval(value)
                lr_scheduler_kwargs[key] = value

        def wrap_check_needless_num_warmup_steps(return_vals):
            if num_warmup_steps is not None and num_warmup_steps != 0:
                raise ValueError(f"{name} does not require num_warmup_steps. Set to 0.")
            return return_vals

        # Custom scheduler from module path
        if args.lr_scheduler_type:
            lr_scheduler_type = args.lr_scheduler_type
            logger.info(f"Using {lr_scheduler_type} | {lr_scheduler_kwargs}")
            if "." not in lr_scheduler_type:
                lr_scheduler_module = torch.optim.lr_scheduler
            else:
                values = lr_scheduler_type.split(".")
                lr_scheduler_module = importlib.import_module(".".join(values[:-1]))
                lr_scheduler_type = values[-1]
            lr_scheduler_class = getattr(lr_scheduler_module, lr_scheduler_type)
            return lr_scheduler_class(optimizer, **lr_scheduler_kwargs)

        if name.lower() == "rex":
            return RexLR(
                optimizer,
                max_lr=args.learning_rate,
                min_lr=(args.learning_rate * min_lr_ratio if min_lr_ratio is not None else args.learning_rate * 0.01),
                num_steps=num_training_steps,
                num_warmup_steps=num_warmup_steps,
                **lr_scheduler_kwargs,
            )

        if name == DiffusersSchedulerType.PIECEWISE_CONSTANT.value:
            name_enum = DiffusersSchedulerType(name)
            schedule_func = DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION[name_enum]
            return schedule_func(optimizer, **lr_scheduler_kwargs)

        name_enum = SchedulerType(name)
        schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name_enum]

        if name_enum == SchedulerType.CONSTANT:
            return wrap_check_needless_num_warmup_steps(schedule_func(optimizer, **lr_scheduler_kwargs))

        if num_warmup_steps is None:
            raise ValueError(f"{name} requires num_warmup_steps")

        if name_enum == SchedulerType.CONSTANT_WITH_WARMUP:
            return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **lr_scheduler_kwargs)

        if name_enum == SchedulerType.INVERSE_SQRT:
            return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, timescale=timescale, **lr_scheduler_kwargs)

        if num_training_steps is None:
            raise ValueError(f"{name} requires num_training_steps")

        if name_enum == SchedulerType.COSINE_WITH_RESTARTS:
            return schedule_func(
                optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps,
                num_cycles=num_cycles, **lr_scheduler_kwargs,
            )

        if name_enum == SchedulerType.POLYNOMIAL:
            return schedule_func(
                optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps,
                power=power, **lr_scheduler_kwargs,
            )

        if name_enum == SchedulerType.COSINE_WITH_MIN_LR:
            return schedule_func(
                optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps,
                num_cycles=num_cycles / 2, min_lr_rate=min_lr_ratio, **lr_scheduler_kwargs,
            )

        if name_enum == SchedulerType.LINEAR or name_enum == SchedulerType.COSINE:
            return schedule_func(
                optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps,
                **lr_scheduler_kwargs,
            )

        if num_decay_steps is None:
            raise ValueError(f"{name} requires num_decay_steps")

        if name_enum == SchedulerType.WARMUP_STABLE_DECAY:
            return schedule_func(
                optimizer, num_warmup_steps=num_warmup_steps, num_stable_steps=num_stable_steps,
                num_decay_steps=num_decay_steps, num_cycles=num_cycles / 2,
                min_lr_ratio=min_lr_ratio if min_lr_ratio is not None else 0.0, **lr_scheduler_kwargs,
            )

        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps,
            num_decay_steps=num_decay_steps, **lr_scheduler_kwargs,
        )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def resume_from_local_if_specified(self, accelerator: Accelerator, args: argparse.Namespace) -> bool:
        """Resume training from a local checkpoint if --resume is specified."""
        if not args.resume:
            return False
        logger.info(f"Resuming training from local state: {args.resume}")
        accelerator.load_state(args.resume)
        return True

    # ------------------------------------------------------------------
    # Timestep sampling
    # ------------------------------------------------------------------

    def get_bucketed_timestep(self) -> float:
        if self.num_timestep_buckets is None or self.num_timestep_buckets <= 1:
            return random.random()

        if len(self.timestep_range_pool) == 0:
            bucket_size = 1.0 / self.num_timestep_buckets
            for i in range(self.num_timestep_buckets):
                self.timestep_range_pool.append((i * bucket_size, (i + 1) * bucket_size))
            random.shuffle(self.timestep_range_pool)

        a, b = self.timestep_range_pool.pop()
        return random.uniform(a, b)

    def get_noisy_model_input_and_timesteps(
        self,
        args: argparse.Namespace,
        noise: torch.Tensor,
        latents: torch.Tensor,
        timesteps: Optional[List[float]],
        noise_scheduler: FlowMatchDiscreteScheduler,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Sample timesteps and create noisy model input via flow matching interpolation."""
        batch_size = noise.shape[0]

        if timesteps is not None:
            timesteps = torch.tensor(timesteps, device=device)

        def uniform_to_normal_ppF(t_uniform: torch.Tensor) -> torch.Tensor:
            eps = 1e-7
            t_uniform = torch.clamp(t_uniform, eps, 1.0 - eps)
            term = 2.0 * t_uniform - 1.0
            return math.sqrt(2.0) * torch.erfinv(term)

        def uniform_to_logsnr_ppF(t_uniform: torch.Tensor, mean: float, std: float) -> torch.Tensor:
            eps = 1e-7
            t_uniform = torch.clamp(t_uniform, eps, 1.0 - eps)
            term = 2.0 * t_uniform - 1.0
            return mean + std * math.sqrt(2.0) * torch.erfinv(term)

        if (
            args.timestep_sampling == "uniform"
            or args.timestep_sampling == "sigmoid"
            or args.timestep_sampling == "shift"
            or args.timestep_sampling == "flux_shift"
            or args.timestep_sampling == "flux2_shift"
            or args.timestep_sampling == "logsnr"
            or args.timestep_sampling == "qinglong_flux"
        ):

            def compute_sampling_timesteps(org_timesteps: Optional[torch.Tensor]) -> torch.Tensor:
                def rand(bs: int, org_ts: Optional[torch.Tensor] = None) -> torch.Tensor:
                    nonlocal device
                    return torch.rand((bs,), device=device) if org_ts is None else org_ts

                def randn(bs: int, org_ts: Optional[torch.Tensor] = None) -> torch.Tensor:
                    nonlocal device
                    return uniform_to_normal_ppF(org_ts) if org_ts is not None else torch.randn((bs,), device=device)

                def rand_logsnr(bs: int, mean: float, std: float, org_ts: Optional[torch.Tensor] = None) -> torch.Tensor:
                    nonlocal device
                    if org_ts is not None:
                        return uniform_to_logsnr_ppF(org_ts, mean, std)
                    return torch.normal(mean=mean, std=std, size=(bs,), device=device)

                if args.timestep_sampling == "uniform" or args.timestep_sampling == "sigmoid":
                    if args.timestep_sampling == "sigmoid":
                        t = torch.sigmoid(args.sigmoid_scale * randn(batch_size, org_timesteps))
                    else:
                        t = rand(batch_size, org_timesteps)

                elif args.timestep_sampling.endswith("shift"):
                    if args.timestep_sampling == "shift":
                        shift = args.discrete_flow_shift
                    else:
                        h, w = latents.shape[-2:]
                        if args.timestep_sampling == "flux_shift":
                            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
                        elif args.timestep_sampling == "flux2_shift":
                            mu = get_lin_function(y1=0.5, y2=1.15)(h * w)
                        shift = math.exp(mu)

                    logits_norm = randn(batch_size, org_timesteps)
                    logits_norm = logits_norm * args.sigmoid_scale
                    t = logits_norm.sigmoid()
                    t = (t * shift) / (1 + (shift - 1) * t)

                elif args.timestep_sampling == "logsnr":
                    logsnr = rand_logsnr(batch_size, args.logit_mean, args.logit_std, org_timesteps)
                    t = torch.sigmoid(-logsnr / 2)

                elif args.timestep_sampling == "qinglong_flux":
                    # Qinglong triple hybrid: mid_shift:logsnr:logsnr2 = 0.80:0.075:0.125
                    decision_t = torch.rand((batch_size,), device=device)
                    mid_mask = decision_t < 0.80
                    logsnr_mask = (decision_t >= 0.80) & (decision_t < 0.875)
                    logsnr_mask2 = decision_t >= 0.875

                    t = torch.zeros((batch_size,), device=device)

                    if mid_mask.any():
                        mid_count = mid_mask.sum().item()
                        h, w = latents.shape[-2:]
                        mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
                        shift = math.exp(mu)
                        logits_norm_mid = randn(mid_count, org_timesteps[mid_mask] if org_timesteps is not None else None)
                        logits_norm_mid = logits_norm_mid * args.sigmoid_scale
                        t_mid = logits_norm_mid.sigmoid()
                        t_mid = (t_mid * shift) / (1 + (shift - 1) * t_mid)
                        t[mid_mask] = t_mid

                    if logsnr_mask.any():
                        logsnr_count = logsnr_mask.sum().item()
                        logsnr = rand_logsnr(
                            logsnr_count, args.logit_mean, args.logit_std,
                            org_timesteps[logsnr_mask] if org_timesteps is not None else None,
                        )
                        t[logsnr_mask] = torch.sigmoid(-logsnr / 2)

                    if logsnr_mask2.any():
                        logsnr2_count = logsnr_mask2.sum().item()
                        logsnr2 = rand_logsnr(
                            logsnr2_count, 5.36, 1.0,
                            org_timesteps[logsnr_mask2] if org_timesteps is not None else None,
                        )
                        t[logsnr_mask2] = torch.sigmoid(-logsnr2 / 2)

                return t  # 0 to 1

            t_min = args.min_timestep if args.min_timestep is not None else 0
            t_max = args.max_timestep if args.max_timestep is not None else 1000.0
            t_min /= 1000.0
            t_max /= 1000.0

            if not args.preserve_distribution_shape:
                t = compute_sampling_timesteps(timesteps)
                t = t * (t_max - t_min) + t_min
            else:
                max_loops = 1000
                available_t = []
                for _ in range(max_loops):
                    ts_input = None
                    if self.num_timestep_buckets is not None:
                        ts_input = torch.tensor([self.get_bucketed_timestep() for _ in range(batch_size)], device=device)
                    t = compute_sampling_timesteps(ts_input)
                    for t_i in t:
                        if t_min <= t_i <= t_max:
                            available_t.append(t_i)
                        if len(available_t) == batch_size:
                            break
                    if len(available_t) == batch_size:
                        break
                if len(available_t) < batch_size:
                    logger.warning(f"Could not sample {batch_size} valid timesteps in {max_loops} loops")
                    t = compute_sampling_timesteps(timesteps)
                else:
                    t = torch.stack(available_t, dim=0)

            timesteps = t * 1000.0
            # Klein latents are 4D: (B, C, H, W) — no video frame dimension
            t = t.view(-1, 1, 1, 1)
            noisy_model_input = (1 - t) * latents + t * noise

            timesteps += 1  # 1 to 1000

        else:
            # sigma-based timestep sampling with weighting schemes
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=batch_size,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            )
            t_min = args.min_timestep if args.min_timestep is not None else 0
            t_max = args.max_timestep if args.max_timestep is not None else 1000
            # min/max are timestep VALUES (0-1000), same as every other sampling mode.
            # The schedule is DESCENDING (timesteps[i] ~= 1000 - i), so convert the value
            # window to an index window first. Previously min/max were used directly as
            # indices, which selected the OPPOSITE end of the schedule (e.g. 0-400 picked
            # timesteps 1000-600). For the default full range this is identical to the
            # old behaviour, including the u-density orientation.
            n = noise_scheduler.timesteps.shape[0]
            lo_idx = n - t_max   # index of the highest requested timestep
            hi_idx = n - t_min   # index of the lowest requested timestep
            indices = (u * (hi_idx - lo_idx) + lo_idx).long().clamp(0, n - 1)

            timesteps = noise_scheduler.timesteps[indices].to(device=device)

            sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=latents.ndim, dtype=dtype)
            noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

        return noisy_model_input, timesteps

    # ------------------------------------------------------------------
    # Klein DiT forward pass
    # ------------------------------------------------------------------

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ):
        """Forward pass through KleinDiT — packs latents, runs the model, computes flow matching target."""
        model: KleinDiT = transformer

        bsize = latents.shape[0]
        packed_latent_height = latents.shape[2]
        packed_latent_width = latents.shape[3]

        # Pack noisy latents to sequence format
        noisy_model_input, img_ids = prc_img(noisy_model_input)  # (B, HW, C), (B, HW, 4)

        # Pack control latents if present
        num_control_images = 0
        ref_tokens, ref_ids = None, None
        if "latents_control_0" in batch:
            control_latents: list[torch.Tensor] = []
            while True:
                key = f"latents_control_{num_control_images}"
                if key in batch:
                    control_latents.append(batch[key])
                    num_control_images += 1
                else:
                    break
            ref_tokens, ref_ids = pack_control_latent(control_latents)

        # Process text context
        ctx_vec = batch["ctx_vec"]  # (B, T, D)
        ctx, ctx_ids = prc_txt(ctx_vec)  # (B, T, D), (B, T, 4)

        # Ensure gradient flow for gradient checkpointing
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            ctx.requires_grad_(True)
            if ref_tokens is not None:
                ref_tokens.requires_grad_(True)

        # Move to device and dtype
        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)
        img_ids = img_ids.to(device=accelerator.device)
        if ref_tokens is not None:
            ref_tokens = ref_tokens.to(device=accelerator.device, dtype=network_dtype)
            ref_ids = ref_ids.to(device=accelerator.device)
        ctx = ctx.to(device=accelerator.device, dtype=network_dtype)
        ctx_ids = ctx_ids.to(device=accelerator.device)

        # Guidance vector — use 1.0 for training (non-base models use embedded guidance)
        guidance_vec = torch.full((bsize,), 1.0, device=accelerator.device, dtype=network_dtype)

        # Concatenate control tokens if present
        img_input = noisy_model_input
        img_input_ids = img_ids
        if ref_tokens is not None:
            img_input = torch.cat((img_input, ref_tokens), dim=1)
            img_input_ids = torch.cat((img_input_ids, ref_ids), dim=1)

        # Klein uses timesteps in [0, 1] range
        timesteps = timesteps / 1000.0

        # Run DiT
        model_pred = model(
            x=img_input, x_ids=img_input_ids, timesteps=timesteps, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance_vec
        )
        model_pred = model_pred[:, : noisy_model_input.shape[1]]  # trim to image tokens only

        # Unpack back to spatial format
        model_pred = rearrange(model_pred, "b (h w) c -> b c h w", h=packed_latent_height, w=packed_latent_width)

        # Flow matching loss: target = noise - latents
        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        target = noise - latents

        return model_pred, target

    # ------------------------------------------------------------------
    # Sample image generation
    # ------------------------------------------------------------------

    def process_sample_prompts(self, args: argparse.Namespace, accelerator: Accelerator, sample_prompts: str):
        """Encode sample prompts with the Qwen3 text encoder."""
        device = accelerator.device

        logger.info(f"Caching Text Encoder outputs for sample prompts: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        # Load text encoder
        te_dtype = torch.float8_e4m3fn if args.fp8_text_encoder else torch.bfloat16
        text_embedder = load_text_encoder(args.text_encoder, dtype=te_dtype, device=device, disable_mmap=True)

        logger.info("Encoding with Qwen3 Text Encoder...")
        sample_prompts_te_outputs = {}
        for prompt_dict in prompts:
            if "negative_prompt" not in prompt_dict:
                prompt_dict["negative_prompt"] = " "

            for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", " ")]:
                if p is None or p in sample_prompts_te_outputs:
                    continue

                logger.info(f"  Encoding: {p[:80]}...")
                with torch.no_grad():
                    if te_dtype.itemsize == 1:
                        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                            ctx_vec = text_embedder([p])
                    else:
                        ctx_vec = text_embedder([p])
                sample_prompts_te_outputs[p] = ctx_vec.cpu()

        del text_embedder
        clean_memory_on_device(device)

        # Build sample parameters
        _global_ref = getattr(args, "sample_ref_image", None)
        _global_ref = _global_ref if (_global_ref and os.path.exists(_global_ref)) else None
        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()

            p = prompt_dict.get("prompt", "")
            prompt_dict_copy["ctx_vec"] = sample_prompts_te_outputs[p]
            p = prompt_dict.get("negative_prompt", " ")
            prompt_dict_copy["negative_ctx_vec"] = sample_prompts_te_outputs[p]

            # Global sample reference image (Samples tab) — unless the prompt line
            # already specified its own control image.
            if _global_ref is not None:
                prompt_dict_copy.setdefault("control_image_path", [_global_ref])

            sample_parameters.append(prompt_dict_copy)

        clean_memory_on_device(accelerator.device)
        return sample_parameters

    def _get_override_sample_params(self, args, accelerator, base_params):
        """Live sample override (written by the GUI to <output_dir>/.sample_override.json).

        While the file exists, samples use its prompt/seed/resolution instead of
        the Samples-tab set. The positive prompt is re-encoded ONLY when its text
        changes (cached on self._override_te_cache) — the negative embedding and
        step settings are inherited from the configured params, so seed/res-only
        tweaks cost nothing. Returns [one sample_parameter] or None (no override).
        """
        import json
        if not base_params:
            return None
        path = os.path.join(args.output_dir, ".sample_override.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                ov = json.load(f)
        except Exception:
            return None
        prompt = str(ov.get("prompt", "")).strip()
        if not prompt:
            return None
        try:
            seed = int(ov.get("seed", base_params[0].get("seed", 42)) or 42)
        except (TypeError, ValueError):
            seed = 42
        try:
            width = (int(ov.get("width", base_params[0].get("width", 768)) or 768) // 16) * 16
            height = (int(ov.get("height", base_params[0].get("height", 768)) or 768) // 16) * 16
        except (TypeError, ValueError):
            width = height = 768

        cache = getattr(self, "_override_te_cache", None)
        if cache is None or cache.get("prompt") != prompt:
            # Re-encode only on a genuine prompt-text change.
            te_dtype = torch.float8_e4m3fn if args.fp8_text_encoder else torch.bfloat16
            text_embedder = load_text_encoder(args.text_encoder, dtype=te_dtype,
                                              device=accelerator.device, disable_mmap=True)
            with torch.no_grad():
                if te_dtype.itemsize == 1:
                    with torch.amp.autocast(device_type=accelerator.device.type, dtype=torch.bfloat16):
                        ctx_vec = text_embedder([prompt])
                else:
                    ctx_vec = text_embedder([prompt])
            cache = {"prompt": prompt, "ctx_vec": ctx_vec.cpu()}
            self._override_te_cache = cache
            del text_embedder
            clean_memory_on_device(accelerator.device)
            logger.info(f"  [sample override] encoded new prompt: {prompt[:70]}")

        param = dict(base_params[0])  # inherit negative_ctx_vec + step/guidance settings
        param["prompt"] = prompt
        param["ctx_vec"] = cache["ctx_vec"]
        param["seed"] = seed
        param["width"] = width
        param["height"] = height
        # Reference image (Klein edit conditioning) — the override owns the ref
        # state, so clear any inherited one and add it only if set + present.
        param.pop("control_image_path", None)
        ref = str(ov.get("ref_image", "") or "").strip()
        if ref and os.path.exists(ref):
            param["control_image_path"] = [ref]
        return [param]

    def _encode_control_latents(self, sample_parameter, vae, device):
        """Build (ref_tokens, ref_ids) from sample_parameter['control_image_path'],
        or (None, None) if none. Used by BOTH the Base (do_inference) and Distilled
        (inline) sample paths so an edit-conditioning reference image works either way.

        Reference images are hard-capped to ~0.20 MP (downscale only) BEFORE the VAE
        encode: the ref is encoded and its tokens are concatenated onto the sample
        sequence, so a big image means a huge encode + huge attention = OOM. Capping
        the area bounds both, whatever size the user drops in. Strength is fixed at
        1.0 (stock Klein conditioning)."""
        paths = sample_parameter.get("control_image_path")
        if not paths:
            return None, None
        _MAX_REF_PIXELS = 200_000
        vae.to(device)
        vae.eval()
        latents = []
        with torch.no_grad():
            for p in paths:
                img = Image.open(p).convert("RGB")
                w_c, h_c = img.size
                if w_c * h_c > _MAX_REF_PIXELS:
                    s = (_MAX_REF_PIXELS / (w_c * h_c)) ** 0.5
                    w_c, h_c = int(w_c * s), int(h_c * s)
                w_c = max(16, (w_c // 16) * 16)
                h_c = max(16, (h_c // 16) * 16)
                img = img.resize((w_c, h_c), Image.LANCZOS)
                t = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
                t = t.permute(2, 0, 1).unsqueeze(0)  # NCHW
                latents.append(vae.encode(t.to(device, vae.dtype)).squeeze(0))
        return pack_control_latent(latents)

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=None,
        control_video_path=None,
        force_guidance_embed=False,
    ):
        """Run Klein denoising loop and decode to pixels."""
        model: KleinDiT = transformer
        device = accelerator.device

        # Get embeddings
        ctx = sample_parameter["ctx_vec"].to(device=device, dtype=torch.bfloat16)
        ctx, ctx_ids = prc_txt(ctx)
        negative_ctx = sample_parameter.get("negative_ctx_vec").to(device=device, dtype=torch.bfloat16)
        negative_ctx, negative_ctx_ids = prc_txt(negative_ctx)

        # Initialize random latents
        packed_latent_height, packed_latent_width = height // 16, width // 16
        latents = randn_tensor(
            (1, 128, packed_latent_height, packed_latent_width),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        x, x_ids = prc_img(latents)

        # Prepare control/reference latents if present (shared with the Distilled path).
        ref_tokens, ref_ids = self._encode_control_latents(sample_parameter, vae, device)
        if ref_tokens is not None:
            vae.to("cpu")
            clean_memory_on_device(device)

        # Denoise
        timesteps = get_schedule(sample_steps, x.shape[1], discrete_flow_shift)
        if self.model_version_info.guidance_distilled or force_guidance_embed:
            x = denoise(
                model, x, x_ids, ctx, ctx_ids,
                timesteps=timesteps, guidance=guidance_scale,
                img_cond_seq=ref_tokens, img_cond_seq_ids=ref_ids,
            )
        else:
            # Klein 9B Base has no guidance embed — CFG is the only way guidance works.
            if do_classifier_free_guidance:
                x = denoise_cfg(
                    model, x, x_ids, ctx, ctx_ids, negative_ctx, negative_ctx_ids,
                    timesteps=timesteps, guidance=cfg_scale,
                    img_cond_seq=ref_tokens, img_cond_seq_ids=ref_ids,
                )
            else:
                # No negative prompt — use denoise() but guidance has no effect on Base
                x = denoise(
                    model, x, x_ids, ctx, ctx_ids,
                    timesteps=timesteps, guidance=guidance_scale,
                    img_cond_seq=ref_tokens, img_cond_seq_ids=ref_ids,
                )

        # Unpack to spatial
        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        latent = x.to(vae.dtype)
        del x

        # Decode latents
        vae.to(device)
        vae.eval()

        logger.info(f"Decoding image from latents: {latent.shape}")
        with torch.no_grad():
            pixels = vae.decode(latent)
        del latent

        logger.info("Decoding complete")
        pixels = pixels.to(torch.float32).cpu()
        pixels = (pixels / 2 + 0.5).clamp(0, 1)

        vae.to("cpu")
        clean_memory_on_device(device)

        # Add dummy video frame dimension: B C H W -> B C 1 H W
        pixels = pixels.unsqueeze(2)
        return pixels

    def _auto_distilled_sample_swap(self, num_double: int, num_single: int):
        """Pick how many blocks of the Distilled sample model to swap, by GPU VRAM.

        The Distilled forward is forward-only (no backward) and runs 4 steps once
        per epoch, so swapping is cheap (~1s) and lets the second model coexist
        with the Base on a tight card. Returns (num_blocks, double_override,
        single_override); overrides are None to let enable_block_swap's ratio
        formula split the count.

        Thresholds account for cards reporting slightly under nominal (a 16GB card
        reads ~15.x GiB, a 24GB card ~23.x) — same convention as the training and
        inference auto-detect:
          >=23 GB  -> 0   (24GB+: Distilled ~9GB fits resident, no swap)
          15-22 GB -> 16  (16/20GB: ratio formula swaps ~24 blocks, ~3GB resident)
          <15 GB   -> max (12GB: 2 double + 2 single resident, ~2GB)
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return 0, None, None
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            return 0, None, None
        if vram_gb >= 23:
            return 0, None, None
        if vram_gb >= 15:
            return 16, None, None
        # <15GB — maximum forward-only swap (the ratio formula can't reach it).
        return (num_double - 2) + (num_single - 2), num_double - 2, num_single - 2

    def _should_cache_sample_model(self, args, sample_blocks: int) -> bool:
        """Whether to keep the Distilled sample model resident in CPU RAM between
        epochs instead of reloading it from disk each sample (~3-4s saved).

        Only when the Distilled is NOT block-swapped (sample_blocks==0 — a roomy
        GPU; swapped = tight card, which also tends to be RAM-tight). Mode comes
        from --cache_sample_model:
          off  → never
          on   → always (power user who knows the RAM is there)
          auto → only when free system RAM has a comfortable margin over the
                 ~10 GB fp8 model (degrades to disk reload — never risks OOM).
        """
        mode = str(getattr(args, "cache_sample_model", "auto") or "auto").lower()
        if mode == "off" or sample_blocks > 0:
            return False
        if mode == "on":
            return True
        # auto: decide ONCE per run and stick with it. The check must be one-shot
        # because once the model is cached it consumes ~10 GB of the very
        # "available" RAM we measure — re-checking each epoch would then see less
        # free RAM, drop the cache, free the RAM, re-cache… a flip-flop on
        # marginal-RAM systems. The first reading (before anything is cached) is
        # the true gate.
        decided = getattr(self, "_cache_auto_decision", None)
        if decided is None:
            try:
                import psutil
                decided = psutil.virtual_memory().available / 1e9 >= 18.0  # ~10 GB model + ~8 GB headroom
            except Exception:
                decided = False  # can't check RAM → safe default: don't cache
            self._cache_auto_decision = decided
        return decided

    def sample_images(self, accelerator: Accelerator, args, epoch, steps, vae, transformer, sample_parameters, dit_dtype):
        """Orchestrate sample image generation at the right training steps."""
        if not should_sample_images(args, steps, epoch):
            return

        logger.info("")
        logger.info(f"Generating sample images at step: {steps}")
        if sample_parameters is None:
            logger.error(f"No prompt file: {args.sample_prompts}")
            return

        # Live sample override (GUI-written sentinel): while active, use its
        # prompt/seed/resolution instead of the configured Samples-tab set.
        try:
            _override_params = self._get_override_sample_params(args, accelerator, sample_parameters)
            if _override_params is not None:
                logger.info("  [sample override] active — using override prompt/seed/res")
                sample_parameters = _override_params
        except Exception:
            logger.warning("  [sample override] failed; using configured prompts", exc_info=True)

        distributed_state = PartialState()
        transformer = accelerator.unwrap_model(transformer)

        # Determine whether to use Distilled model for sampling
        _sample_dit_path = getattr(args, "sample_dit", None)
        _use_distilled = _sample_dit_path and os.path.exists(_sample_dit_path)
        _orig_blocks_to_swap = self.blocks_to_swap or 0
        # A 4-bit (NF4) base can't be block-swapped (weights are in _nf4_packed, not
        # .weight), and it's already small (~5.6 GB), so we skip the swap-maxing that
        # the fp8/bf16 path does to free VRAM for the Distilled load.
        _nf4_base = getattr(transformer, "_nf4_quantized", False)
        _distilled_dit = None
        _sample_net = None
        _ctx_sample = None

        if _use_distilled:
            logger.info(f"Using Distilled model for samples: {_sample_dit_path}")
            # Max block-swap Base DiT to free VRAM for the second (Distilled) model.
            # Klein 9B: 8 double + 24 single. True max swap = (8-2) + (24-2) = 28,
            # leaving only 2 double + 2 single resident. The ratio formula caps at 24
            # (it locks single = 3x double, so 6 double + 18 single), so pass the
            # per-type counts explicitly to also swap the last 4 single blocks
            # (~1 GB more) — matters on a tight 16 GB peak where Base + Distilled
            # both sit on the GPU.
            if not _nf4_base:
                _max_double = getattr(transformer, "num_double_blocks", 8) - 2
                _max_single = getattr(transformer, "num_single_blocks", 24) - 2
                transformer.enable_block_swap(
                    _max_double + _max_single, device=accelerator.device, supports_backward=True,
                    double_blocks_to_swap=_max_double, single_blocks_to_swap=_max_single,
                )
                transformer.prepare_block_swap_before_forward()
            clean_memory_on_device(accelerator.device)

            # Load separate Distilled DiT — same path as Explorer/Repair Studio.
            # Decide how many of ITS blocks to swap. The Distilled forward is
            # forward-only (no backward) and runs 4 steps once per epoch, so we can
            # swap freely to keep its resident footprint small while the Base is
            # also on the GPU. An explicit --sample_blocks_to_swap wins; otherwise
            # auto-pick by VRAM.
            from fizgig.klein.model_utils import load_dit, KLEIN_MODEL_INFO
            _nd = getattr(transformer, "num_double_blocks", 8)
            _ns = getattr(transformer, "num_single_blocks", 24)
            _arg_blocks = getattr(args, "sample_blocks_to_swap", 0) or 0
            if _arg_blocks > 0:
                _sample_blocks, _samp_d, _samp_s = _arg_blocks, None, None
            else:
                _sample_blocks, _samp_d, _samp_s = self._auto_distilled_sample_swap(_nd, _ns)
            if _sample_blocks > 0:
                logger.info(f"Distilled sample block swap: {_sample_blocks} "
                            f"(double={_samp_d if _samp_d is not None else 'auto'}, "
                            f"single={_samp_s if _samp_s is not None else 'auto'})")
            from fizgig.networks.lora_klein import create_arch_network_from_weights as _sample_create
            from fizgig.networks.lora import ensure_kohya_lora_state_dict as _sample_ensure
            from fizgig.training.train_utils import get_epoch_ckpt_name as _get_ckpt
            from safetensors.torch import load_file as _sample_load_lora
            _ckpt_path = os.path.join(args.output_dir, _get_ckpt(args.output_name, epoch))

            # Optionally reuse a CPU-cached Distilled model (skips the disk reload).
            # Only when it isn't block-swapped (sample_blocks==0). Any failure falls
            # back to a fresh disk load, so the cache can never break a run.
            _cache_enabled = self._should_cache_sample_model(args, _sample_blocks)
            _cache = getattr(self, "_distilled_cache", None)
            _reuse = bool(_cache_enabled and _cache is not None
                          and _cache.get("path") == _sample_dit_path)
            if _reuse:
                try:
                    _distilled_dit = _cache["dit"]
                    _sample_net = _cache["net"]
                    _ctx_sample = _cache["ctx"]
                    _distilled_dit.to(accelerator.device).eval()
                    if _ctx_sample is not None:
                        _ctx_sample.to(accelerator.device, dtype=torch.bfloat16).eval()
                    # Refresh the trainable LoRA to THIS epoch's checkpoint.
                    if epoch > 0 and os.path.exists(_ckpt_path):
                        _lora_sd = _sample_ensure(_sample_load_lora(_ckpt_path))
                        if _sample_net is None:
                            # First checkpoint after an epoch-0 (no-LoRA) cache: build+patch now.
                            _sample_net = _sample_create(multiplier=1.0, weights_sd=_lora_sd,
                                                         unet=_distilled_dit, for_inference=True)
                            _sample_net.apply_to(text_encoders=None, unet=_distilled_dit,
                                                 apply_text_encoder=False, apply_unet=True)
                        _sample_net.load_state_dict(_lora_sd, strict=False)
                        _sample_net.to(accelerator.device, dtype=torch.bfloat16).eval()
                    elif _sample_net is not None:
                        _sample_net.to(accelerator.device, dtype=torch.bfloat16).eval()
                    logger.info("  Distilled reused from RAM cache (skipped disk reload)")
                except Exception:
                    logger.warning("  Distilled cache reuse failed — reloading from disk", exc_info=True)
                    self._distilled_cache = None
                    _reuse = False

            if not _reuse:
                _loading_device = "cpu" if _sample_blocks > 0 else accelerator.device
                _distilled_dit = load_dit(
                    device=accelerator.device,
                    model_version_info=KLEIN_MODEL_INFO["klein-9b"],
                    dit_path=_sample_dit_path,
                    attn_mode="torch",
                    split_attn=False,
                    loading_device=_loading_device,
                    dit_weight_dtype=torch.bfloat16,
                    fp8_scaled=False,
                )
                if getattr(args, "sample_int8", False):
                    # INT8 (W8A8) fast preview matmul — quantize the Distilled block Linears BEFORE
                    # block swap (so the offloader stages int8) and before the LoRA wraps them. On
                    # the load device so a swapped (CPU-loaded) sample model doesn't need the whole
                    # int8 model resident on GPU. Cached below, so later epochs reuse the int8 model.
                    from fizgig.modules.int8 import apply_int8_quantization
                    from fizgig.klein.model import FP8_OPTIMIZATION_TARGET_KEYS, FP8_OPTIMIZATION_EXCLUDE_KEYS
                    apply_int8_quantization(_distilled_dit,
                                            target_keys=FP8_OPTIMIZATION_TARGET_KEYS,
                                            exclude_keys=FP8_OPTIMIZATION_EXCLUDE_KEYS,
                                            compute_device=torch.device(_loading_device))
                if _sample_blocks > 0:
                    _distilled_dit.enable_block_swap(
                        _sample_blocks, device=accelerator.device, supports_backward=False,
                        double_blocks_to_swap=_samp_d, single_blocks_to_swap=_samp_s,
                    )
                    _distilled_dit.move_to_device_except_swap_blocks(accelerator.device)
                _distilled_dit.prepare_block_swap_before_forward()
                _distilled_dit.eval()

                # Apply trainable LoRA from saved checkpoint — matches Repair Studio's loading path
                if epoch > 0 and os.path.exists(_ckpt_path):
                    logger.info(f"  Loading trainable LoRA from disk: {_ckpt_path}")
                    _lora_sd = _sample_ensure(_sample_load_lora(_ckpt_path))
                    _sample_net = _sample_create(
                        multiplier=1.0, weights_sd=_lora_sd, unet=_distilled_dit, for_inference=True,
                    )
                    _sample_net.apply_to(text_encoders=None, unet=_distilled_dit,
                                         apply_text_encoder=False, apply_unet=True)
                    _sample_net.load_state_dict(_lora_sd, strict=False)
                    _sample_net.to(accelerator.device, dtype=torch.bfloat16)
                    _sample_net.eval()
                else:
                    logger.info(f"  Epoch {epoch}: no checkpoint yet, sampling without trainable LoRA")

                # Apply context LoRA to Distilled DiT if active
                logger.info(f"  Context LoRA check: self.context_network is {'set' if self.context_network is not None else 'None'}")
                if self.context_network is not None:
                    # Context LoRA weights are frozen — get them from the stored network
                    _ctx_sd = self.context_network.state_dict()
                    _ctx_sd = {k: v.cpu().to(torch.bfloat16) for k, v in _ctx_sd.items()}
                    _ctx_sample = _sample_create(
                        multiplier=self.context_network.multiplier,
                        weights_sd=_ctx_sd, unet=_distilled_dit, for_inference=True,
                    )
                    _ctx_sample.apply_to(text_encoders=None, unet=_distilled_dit,
                                         apply_text_encoder=False, apply_unet=True)
                    _ctx_sample.load_state_dict(_ctx_sd, strict=False)
                    _ctx_sample.to(accelerator.device, dtype=torch.bfloat16)
                    _ctx_sample.eval()

            sample_transformer = _distilled_dit
            logger.info("  Distilled loaded — sampling with 4 steps, guidance 1.0")
        else:
            transformer.switch_block_swap_for_inference()
            sample_transformer = transformer

        save_dir = os.path.join(args.output_dir, "sample")
        os.makedirs(save_dir, exist_ok=True)

        # Save random state
        rng_state = torch.get_rng_state()
        cuda_rng_state = None
        try:
            cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        except Exception:
            pass

        if distributed_state.num_processes <= 1:
            # Distilled path manages its own no_grad/autocast per step (matching Repair Studio).
            # Base path still uses accelerator.autocast().
            if _use_distilled:
                for sample_parameter in sample_parameters:
                    self.sample_image_inference(
                        accelerator, args, sample_transformer, dit_dtype, vae, save_dir,
                        sample_parameter, epoch, steps, use_distilled=True,
                    )
                    clean_memory_on_device(accelerator.device)
            else:
                with torch.no_grad(), accelerator.autocast():
                    for sample_parameter in sample_parameters:
                        self.sample_image_inference(
                            accelerator, args, sample_transformer, dit_dtype, vae, save_dir,
                            sample_parameter, epoch, steps, use_distilled=False,
                        )
                        clean_memory_on_device(accelerator.device)
        else:
            per_process_params = []
            for i in range(distributed_state.num_processes):
                per_process_params.append(sample_parameters[i :: distributed_state.num_processes])

            with torch.no_grad():
                with distributed_state.split_between_processes(per_process_params) as sample_parameter_lists:
                    for sample_parameter in sample_parameter_lists[0]:
                        self.sample_image_inference(
                            accelerator, args, sample_transformer, dit_dtype, vae, save_dir,
                            sample_parameter, epoch, steps, use_distilled=_use_distilled,
                        )
                        clean_memory_on_device(accelerator.device)

        # Restore random state
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)

        if _use_distilled:
            if _cache_enabled:
                # Keep the Distilled (+ its LoRA networks) resident in CPU RAM for
                # the next epoch's sample instead of deleting it — that's the win:
                # no disk reload next time. Move off GPU to free training VRAM.
                try:
                    _distilled_dit.to("cpu")
                    if _sample_net is not None:
                        _sample_net.to("cpu")
                    if _ctx_sample is not None:
                        _ctx_sample.to("cpu")
                    self._distilled_cache = {
                        "dit": _distilled_dit, "net": _sample_net,
                        "ctx": _ctx_sample, "path": _sample_dit_path,
                    }
                except Exception:
                    logger.warning("  Failed to cache Distilled to RAM; will reload next time", exc_info=True)
                    self._distilled_cache = None
            else:
                # Unload Distilled DiT — break LoRA circular references first
                for _net in (_sample_net, _ctx_sample):
                    if _net is not None:
                        for lora in getattr(_net, 'unet_loras', []):
                            if hasattr(lora, 'org_forward'):
                                lora.org_forward = None
                        if hasattr(_net, 'unet_loras'):
                            _net.unet_loras.clear()
                del _sample_net, _ctx_sample
                _sample_net = _ctx_sample = None
                # Move Distilled to CPU before deleting to free GPU immediately
                _distilled_dit.to("cpu")
                del _distilled_dit
                _distilled_dit = None
            import gc; gc.collect(); gc.collect()
            torch.cuda.empty_cache()
            clean_memory_on_device(accelerator.device)
            # Restore Base DiT to training state
            if _orig_blocks_to_swap > 0:
                # Re-enable original block swap level
                transformer.enable_block_swap(_orig_blocks_to_swap, device=accelerator.device, supports_backward=True)
                transformer.move_to_device_except_swap_blocks(accelerator.device)
            elif not _nf4_base:
                # No block swap during training — tear down the sample-time swap
                # offloaders (their backward hooks would otherwise linger and
                # corrupt the next backward) and move everything back to GPU.
                transformer.disable_block_swap()
                transformer.to(accelerator.device)
            # (NF4 base: we never swapped it during sampling, so nothing to restore.)
            transformer.switch_block_swap_for_training()
        else:
            transformer.switch_block_swap_for_training()
        clean_memory_on_device(accelerator.device)

    def sample_image_inference(self, accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps, use_distilled=False):
        """Generate a single sample image for a given prompt."""
        if use_distilled:
            # Distilled: 4 steps, guidance embed, no CFG — matches Explorer/Repair Studio
            sample_steps = 4
            guidance_scale = 1.0
            discrete_flow_shift = None  # dynamic empirical mu
        else:
            default_steps = int(self.model_version_info.defaults.get("num_steps", 20))
            sample_steps = sample_parameter.get("sample_steps", default_steps)
            guidance_scale = sample_parameter.get("guidance_scale", self.default_guidance_scale)
            discrete_flow_shift = sample_parameter.get("discrete_flow_shift", self.default_discrete_flow_shift)

        width = sample_parameter.get("width", 256)
        height = sample_parameter.get("height", 256)
        frame_count = 1  # Klein is image-only
        seed = sample_parameter.get("seed")
        prompt: str = sample_parameter.get("prompt", "")
        cfg_scale = sample_parameter.get("cfg_scale", None)
        negative_prompt = sample_parameter.get("negative_prompt", None)

        # Round to multiples of 8
        width = (width // 8) * 8
        height = (height // 8) * 8

        image_path = None  # Klein does not have I2V mode

        device = accelerator.device
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            torch.seed()
            torch.cuda.seed()
            generator = torch.Generator(device=device).manual_seed(torch.initial_seed())

        logger.info(f"  prompt: {prompt}")
        logger.info(f"  size: {width}x{height}")
        mode_str = "Distilled 4-step" if use_distilled else "Base"
        logger.info(f"  mode: {mode_str}, steps: {sample_steps}, guidance: {guidance_scale}, shift: {discrete_flow_shift}")
        if seed is not None:
            logger.info(f"  seed: {seed}")

        if use_distilled:
            do_classifier_free_guidance = False
        else:
            do_classifier_free_guidance = False
            if negative_prompt is not None:
                do_classifier_free_guidance = True
                logger.info(f"  negative: {negative_prompt}")
                logger.info(f"  cfg_scale: {cfg_scale}")

        # Run inference
        has_self_ref_orig_mod = getattr(transformer, "_orig_mod", None) is transformer
        was_train = transformer.training if not has_self_ref_orig_mod else True
        if not has_self_ref_orig_mod:
            transformer.eval()

        if use_distilled:
            # Inline Distilled denoising — single cond pass, no CFG.
            # Matches Repair Studio exactly. CFG=1.0 dual-pass is NOT equivalent
            # when LoRA is active — the LoRA contributes to both cond and uncond,
            # so CFG partially cancels the LoRA effect.
            from diffusers.utils.torch_utils import randn_tensor
            from fizgig.klein.model_utils import get_simple_euler_schedule
            from fizgig.klein.position import prc_img, prc_txt, scatter_ids

            ctx = sample_parameter["ctx_vec"].to(device=device, dtype=torch.bfloat16)
            ctx, ctx_ids = prc_txt(ctx)

            packed_h, packed_w = height // 16, width // 16
            latents = randn_tensor(
                (1, 128, packed_h, packed_w),
                generator=generator, device=device, dtype=torch.bfloat16,
            )
            x, x_ids = prc_img(latents)

            # Reference image (Klein edit conditioning) — encode + pack, then
            # concatenate onto the cond pass (positive only). Capped to ~0.20 MP
            # inside the helper so it can't OOM the sample. VAE goes back to CPU
            # before the denoise; it's reloaded for decode below.
            _ref_tokens, _ref_ids = self._encode_control_latents(sample_parameter, vae, device)
            _has_ref = _ref_tokens is not None
            if _has_ref:
                vae.to("cpu")
                clean_memory_on_device(device)

            if hasattr(transformer, 'prepare_block_swap_before_forward'):
                transformer.prepare_block_swap_before_forward()

            # Match ComfyUI's Euler Simple schedule (shift=2.02, evenly-spaced)
            timesteps = get_simple_euler_schedule(sample_steps)
            guidance_vec = torch.full((x.shape[0],), guidance_scale, device=device, dtype=x.dtype)

            for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
                t_vec = torch.full((x.shape[0],), t_curr, dtype=x.dtype, device=device)
                if _has_ref:
                    x_fwd = torch.cat((x, _ref_tokens), dim=1)
                    x_fwd_ids = torch.cat((x_ids, _ref_ids), dim=1)
                else:
                    x_fwd, x_fwd_ids = x, x_ids
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=x.dtype):
                    pred = transformer(
                        x=x_fwd, x_ids=x_fwd_ids, timesteps=t_vec,
                        ctx=ctx, ctx_ids=ctx_ids, guidance=guidance_vec,
                    )
                if _has_ref:
                    pred = pred[:, : x.shape[1]]
                x = x + (t_prev - t_curr) * pred

            # Unpack and decode
            x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
            latent = x.to(vae.dtype)
            del x

            vae.to(device)
            vae.eval()
            with torch.no_grad():
                pixels = vae.decode(latent)
            del latent
            pixels = pixels.to(torch.float32).cpu()
            pixels = (pixels / 2 + 0.5).clamp(0, 1)
            vae.to("cpu")
            clean_memory_on_device(device)

            video = pixels.unsqueeze(2)  # B C H W -> B C 1 H W
        else:
            video = self.do_inference(
                accelerator, args, sample_parameter, vae, dit_dtype, transformer,
                discrete_flow_shift, sample_steps, width, height, frame_count, generator,
                do_classifier_free_guidance, guidance_scale, cfg_scale, image_path=image_path,
            )

        if not has_self_ref_orig_mod:
            transformer.train(was_train)

        if video is None:
            logger.error("No image generated")
            return

        # Save the image
        ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
        seed_suffix = "" if seed is None else f"_{seed}"
        prompt_idx = sample_parameter.get("enum", 0)
        save_path = (
            f"{'' if args.output_name is None else args.output_name + '_'}"
            f"{num_suffix}_{prompt_idx:02d}_{ts_str}{seed_suffix}"
        )

        # wandb logging
        wandb_tracker = None
        try:
            wandb_tracker = accelerator.get_tracker("wandb")
            try:
                import wandb
            except ImportError:
                wandb_tracker = None
                wandb = None
        except Exception:
            wandb = None

        # Klein always produces images (video.shape[2] == 1)
        # Convert tensor to PIL and save
        for batch_idx in range(video.shape[0]):
            img_tensor = video[batch_idx, :, 0, :, :]  # C, H, W
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)
            img_path = os.path.join(save_dir, save_path + ".png")
            pil_image.save(img_path)
            logger.info(f"  Saved sample: {img_path}")
            # Tracked so a checkpoint saved after this can default modelspec.description to
            # the prompt that made its thumbnail, instead of shipping blank — same idea as the
            # thumbnail itself, which is already whatever sample happens to be newest on disk.
            self._last_sample_prompt = prompt

            if wandb_tracker is not None and wandb is not None:
                wandb_tracker.log({f"sample_{prompt_idx}": wandb.Image(img_path)}, step=steps)

        vae.to("cpu")
        clean_memory_on_device(accelerator.device)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, args):
        """The main training entry point."""

        # CUDA optimizations
        if torch.cuda.is_available():
            if args.cuda_allow_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled TF32 on CUDA")
            if args.cuda_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
                logger.info("Enabled cuDNN benchmark")

        # Validate required arguments
        if args.dataset_config is None:
            raise ValueError("dataset_config is required")
        if args.dit is None:
            raise ValueError("path to DiT model is required")
        assert not args.fp8_scaled or args.fp8_base, "fp8_scaled requires fp8_base"

        if args.sage_attn:
            raise ValueError("SageAttention does not support training. Use --sdpa or --xformers instead.")

        if args.disable_numpy_memmap:
            logger.info("Disabled numpy memory mapping for model loading")

        # Model-specific setup
        self.handle_model_specific_args(args)

        # Show timesteps mode (debug)
        if args.show_timesteps:
            self._show_timesteps(args)
            return

        session_id = random.randint(0, 2**32)
        training_started_at = time.time()

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        # Timestep bucketing
        if args.num_timestep_buckets is not None:
            logger.info(f"Using timestep bucketing: {args.num_timestep_buckets} buckets")
        self.num_timestep_buckets = args.num_timestep_buckets

        current_epoch = Value("i", 0)

        # Load dataset
        blueprint_generator = BlueprintGenerator(ConfigSanitizer())
        logger.info(f"Loading dataset config from {args.dataset_config}")
        user_config = load_user_config(args.dataset_config)
        blueprint = blueprint_generator.generate(user_config, args, architecture=self.architecture)
        train_dataset_group = generate_dataset_group_by_blueprint(
            blueprint.dataset_group, training=True, num_timestep_buckets=self.num_timestep_buckets, shared_epoch=current_epoch
        )

        if train_dataset_group.num_train_items == 0:
            raise ValueError(
                "No training items found. Ensure latent/Text Encoder caches have been created."
            )

        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = _Collator(current_epoch, ds_for_collator)

        # Prepare accelerator
        logger.info("Preparing accelerator")
        accelerator = prepare_accelerator(args)
        if args.mixed_precision is None:
            args.mixed_precision = accelerator.mixed_precision
            logger.info(f"mixed_precision set to {args.mixed_precision}")
        is_main_process = accelerator.is_main_process

        # Dtype setup
        weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif args.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        dit_dtype = torch.bfloat16 if args.dit_dtype is None else _str_to_dtype(args.dit_dtype)
        dit_weight_dtype = (None if args.fp8_scaled else torch.float8_e4m3fn) if args.fp8_base else dit_dtype
        logger.info(f"DiT precision: {dit_dtype}, weight precision: {dit_weight_dtype}")

        # Sample prompt encoding (before loading DiT to save GPU memory)
        vae_dtype = torch.float32 if args.vae_dtype is None else _str_to_dtype(args.vae_dtype)
        sample_parameters = None
        vae = None
        if args.sample_prompts:
            sample_parameters = self.process_sample_prompts(args, accelerator, args.sample_prompts)
            vae = self.load_vae_model(args, vae_dtype=vae_dtype, vae_path=args.vae)
            vae.requires_grad_(False)
            vae.eval()

        # Load DiT
        blocks_to_swap = args.blocks_to_swap if args.blocks_to_swap else 0
        if getattr(args, "quant_4bit", False) and blocks_to_swap > 0:
            # NF4 weights live in module._nf4_packed, not module.weight (which is
            # freed), so the block-swap machinery would shuffle empty tensors.
            # 4-bit already fits without swap — force it off.
            logger.warning("4-bit base mode: block swap is incompatible — forcing blocks_to_swap=0.")
            blocks_to_swap = 0
        self.blocks_to_swap = blocks_to_swap
        loading_device = "cpu" if blocks_to_swap > 0 else accelerator.device

        logger.info(f"Loading DiT model from {args.dit}")
        if args.sdpa:
            attn_mode = "torch"
        elif args.flash_attn:
            attn_mode = "flash"
        elif args.sage_attn:
            attn_mode = "sageattn"
        elif args.xformers:
            attn_mode = "xformers"
        elif args.flash3:
            attn_mode = "flash3"
        else:
            raise ValueError("One of --sdpa, --flash_attn, --flash3, --sage_attn, or --xformers must be specified")

        transformer = self.load_transformer(
            accelerator, args, args.dit, attn_mode, args.split_attn, loading_device, dit_weight_dtype
        )
        transformer.eval()
        transformer.requires_grad_(False)

        if blocks_to_swap > 0:
            logger.info(
                f"Enabling block swap: {blocks_to_swap} blocks to CPU, "
                f"pinned_memory={args.use_pinned_memory_for_block_swap}"
            )
            transformer.enable_block_swap(
                blocks_to_swap, accelerator.device, supports_backward=True,
                use_pinned_memory=args.use_pinned_memory_for_block_swap,
            )
            transformer.move_to_device_except_swap_blocks(accelerator.device)

        # Load network module (LoRA)
        sys.path.append(os.path.dirname(__file__))
        accelerator.print("Importing network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)

        if args.base_weights is not None:
            for i, weight_path in enumerate(args.base_weights):
                multiplier = 1.0
                if args.base_weights_multiplier is not None and len(args.base_weights_multiplier) > i:
                    multiplier = args.base_weights_multiplier[i]

                accelerator.print(f"Merging base weights: {weight_path} x {multiplier}")
                weights_sd = load_file(weight_path)
                module = network_module.create_arch_network_from_weights(
                    multiplier, weights_sd, unet=transformer, for_inference=True
                )
                module.merge_to(None, transformer, weights_sd, weight_dtype, "cpu")

            accelerator.print(f"All base weights merged: {', '.join(args.base_weights)}")

        # Create network
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                if "=" not in net_arg:
                    raise ValueError(f"--network_args entry {net_arg!r} is not key=value")
                # maxsplit=1: values may legitimately contain '=' — the old unbounded
                # split raised 'too many values to unpack'.
                key, value = net_arg.split("=", 1)
                net_kwargs[key] = value

        if args.dim_from_weights:
            logger.info(f"Loading network dims from weights: {args.dim_from_weights}")
            weights_sd = load_file(args.dim_from_weights)
            network, _ = network_module.create_arch_network_from_weights(1, weights_sd, unet=transformer)
        else:
            if hasattr(network_module, "create_arch_network"):
                network = network_module.create_arch_network(
                    1.0, args.network_dim, args.network_alpha, vae, None, transformer,
                    neuron_dropout=args.network_dropout, **net_kwargs,
                )
            else:
                # LyCORIS compatibility
                network = network_module.create_network(
                    1.0, args.network_dim, args.network_alpha, vae, None, transformer, **net_kwargs,
                )

        if network is None:
            return

        if hasattr(network, "prepare_network"):
            network.prepare_network(args)

        # Context LoRA — load + apply BEFORE the trainable network so the trainable
        # network's wrapped forward includes the context LoRA's contribution. Frozen
        # (no grads), kept on the model throughout training.
        self.context_network = None
        if getattr(args, "context_lora_path", None):
            from safetensors.torch import load_file as _ctx_load_file
            from fizgig.networks.lora_klein import create_arch_network_from_weights as _ctx_create
            from fizgig.networks.lora import ensure_kohya_lora_state_dict as _ctx_ensure_kohya
            logger.info(
                f"Loading context LoRA: {args.context_lora_path} @ strength {args.context_lora_strength}"
            )
            ctx_weights = _ctx_load_file(args.context_lora_path)
            # Auto-convert PEFT/diffusers-format Context LoRAs to Fizgig/kohya format
            ctx_weights = _ctx_ensure_kohya(ctx_weights)
            self.context_network = _ctx_create(
                multiplier=float(args.context_lora_strength),
                weights_sd=ctx_weights,
                unet=transformer,
                for_inference=True,
            )
            self.context_network.apply_to(
                text_encoders=None, unet=transformer,
                apply_text_encoder=False, apply_unet=True,
            )
            ctx_info = self.context_network.load_state_dict(ctx_weights, strict=False)
            logger.info(
                f"  context LoRA loaded: {len(ctx_info.missing_keys)} missing, "
                f"{len(ctx_info.unexpected_keys)} unexpected keys"
            )
            self.context_network.to(accelerator.device, dtype=dit_dtype)
            self.context_network.requires_grad_(False)
            self.context_network.eval()

        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)

        if args.network_weights is not None:
            info = network.load_weights(args.network_weights)
            accelerator.print(f"Loaded network weights from {args.network_weights}: {info}")

        if args.gradient_checkpointing:
            transformer.enable_gradient_checkpointing(args.gradient_checkpointing_cpu_offload)
            network.enable_gradient_checkpointing()

        # Optimizer and dataloader
        accelerator.print("Preparing optimizer, data loader, etc.")

        trainable_params, lr_descriptions = network.prepare_optimizer_params(unet_lr=args.learning_rate)
        optimizer_name, optimizer_args_str, optimizer, optimizer_train_fn, optimizer_eval_fn = self.get_optimizer(
            args, trainable_params
        )

        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        # Calculate training steps
        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
            )
            accelerator.print(f"Steps for {args.max_train_epochs} epochs: {args.max_train_steps}")

        train_dataset_group.set_max_train_steps(args.max_train_steps)

        # LR scheduler
        lr_scheduler = self.get_lr_scheduler(args, optimizer, accelerator.num_processes)

        # Network dtype
        network_dtype = torch.float32

        if dit_weight_dtype != dit_dtype and dit_weight_dtype is not None:
            logger.info(f"Casting model to {dit_weight_dtype}")
            transformer.to(dit_weight_dtype)

        # Prepare with accelerator
        if blocks_to_swap > 0:
            transformer = accelerator.prepare(transformer, device_placement=[not blocks_to_swap > 0])
            accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)
            accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
        else:
            transformer = accelerator.prepare(transformer)

        if args.compile:
            transformer = self.compile_transformer(args, transformer)
            transformer.__dict__["_orig_mod"] = transformer

        network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            network, optimizer, train_dataloader, lr_scheduler
        )
        training_model = network
        self._sample_network_ref = accelerator.unwrap_model(network)  # unwrapped for Distilled sampling

        if args.gradient_checkpointing:
            transformer.train()
        else:
            transformer.eval()

        accelerator.unwrap_model(network).prepare_grad_etc(transformer)

        # Save/load hooks — only save network weights
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                remove_indices = []
                for i, model in enumerate(models):
                    if not isinstance(model, type(accelerator.unwrap_model(network))):
                        remove_indices.append(i)
                for i in reversed(remove_indices):
                    if len(weights) > i:
                        weights.pop(i)

        def load_model_hook(models, input_dir):
            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, type(accelerator.unwrap_model(network))):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                models.pop(i)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        # Resume
        self.resume_from_local_if_specified(accelerator, args)

        # Training calculations
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        accelerator.print("Starting training")
        accelerator.print(f"  Training items: {train_dataset_group.num_train_items}")
        accelerator.print(f"  Batches per epoch: {len(train_dataloader)}")
        accelerator.print(f"  Epochs: {num_train_epochs}")
        accelerator.print(
            f"  Batch size: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
        )
        accelerator.print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
        accelerator.print(f"  Total optimization steps: {args.max_train_steps}")

        # Build metadata
        metadata = {
            "ss_session_id": session_id,
            "ss_training_started_at": training_started_at,
            "ss_output_name": args.output_name,
            "ss_learning_rate": args.learning_rate,
            "ss_num_train_items": train_dataset_group.num_train_items,
            "ss_num_batches_per_epoch": len(train_dataloader),
            "ss_num_epochs": num_train_epochs,
            "ss_gradient_checkpointing": args.gradient_checkpointing,
            "ss_gradient_checkpointing_cpu_offload": args.gradient_checkpointing_cpu_offload,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            SS_METADATA_KEY_BASE_MODEL_VERSION: self.architecture_full_name,
            SS_METADATA_KEY_NETWORK_MODULE: args.network_module,
            SS_METADATA_KEY_NETWORK_DIM: args.network_dim,
            SS_METADATA_KEY_NETWORK_ALPHA: args.network_alpha,
            "ss_network_dropout": args.network_dropout,
            "ss_mixed_precision": args.mixed_precision,
            "ss_seed": args.seed,
            "ss_training_comment": args.training_comment,
            "ss_optimizer": optimizer_name + (f"({optimizer_args_str})" if len(optimizer_args_str) > 0 else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_fp8_base": bool(args.fp8_base),
            "ss_full_fp16": False,
            "ss_full_bf16": False,
            "ss_weighting_scheme": args.weighting_scheme,
            "ss_logit_mean": args.logit_mean,
            "ss_logit_std": args.logit_std,
            "ss_mode_scale": args.mode_scale,
            "ss_guidance_scale": args.guidance_scale,
            "ss_timestep_sampling": args.timestep_sampling,
            "ss_sigmoid_scale": args.sigmoid_scale,
            "ss_discrete_flow_shift": args.discrete_flow_shift,
        }

        # Context LoRA — record so future users know how to deploy this LoRA
        if getattr(args, "context_lora_path", None):
            import os as _os
            metadata["ss_context_lora"] = _os.path.basename(args.context_lora_path)
            metadata["ss_context_lora_strength"] = str(args.context_lora_strength)

        datasets_metadata = []
        for dataset in train_dataset_group.datasets:
            datasets_metadata.append(dataset.get_metadata())
        metadata["ss_datasets"] = json.dumps(datasets_metadata)

        if args.network_args:
            metadata[SS_METADATA_KEY_NETWORK_ARGS] = json.dumps(net_kwargs)

        # Model name metadata
        if args.dit is not None:
            sd_model_name = args.dit
            if os.path.exists(sd_model_name):
                sd_model_name = os.path.basename(sd_model_name)
            metadata["ss_sd_model_name"] = sd_model_name

        if args.vae is not None:
            vae_name = args.vae
            if os.path.exists(vae_name):
                vae_name = os.path.basename(vae_name)
            metadata["ss_vae_name"] = vae_name

        metadata = {k: str(v) for k, v in metadata.items()}

        minimum_metadata = {}
        for key in SS_METADATA_MINIMUM_KEYS:
            if key in metadata:
                minimum_metadata[key] = metadata[key]

        # Init trackers
        if accelerator.is_main_process:
            init_kwargs = {}
            if args.wandb_run_name:
                init_kwargs["wandb"] = {"name": args.wandb_run_name}
            if args.log_tracker_config is not None:
                init_kwargs = toml.load(args.log_tracker_config)
            accelerator.init_trackers(
                "network_train" if args.log_tracker_name is None else args.log_tracker_name,
                config=get_sanitized_config_or_none(args),
                init_kwargs=init_kwargs,
            )

        epoch_to_start = 0
        global_step = 0
        # Resume bug fix: derive epoch + global_step from the --resume state directory name.
        # State dirs are named "<output_name>-NNNNNN-state" (6-digit zero-padded epoch number).
        if args.resume:
            import re as _re
            _resume_basename = os.path.basename(args.resume.rstrip(os.sep).rstrip("/"))
            _m = _re.search(r'-(\d{6})-state', _resume_basename)
            if _m:
                epoch_to_start = int(_m.group(1))
                # global_step counts OPTIMIZER steps (incremented on sync_gradients), and
                # max_train_steps is in the same unit — seeding from raw micro-batches
                # inflated it by the accumulation factor, so a resumed run with accum >= 2
                # hit `global_step >= max_train_steps` immediately and trained nothing
                # while still writing checkpoints and samples.
                global_step = epoch_to_start * num_update_steps_per_epoch
                accelerator.print(
                    f"[resume] resuming from epoch {epoch_to_start}, global_step {global_step}"
                )
                import sys as _sys; _sys.stdout.flush()
            else:
                accelerator.print(
                    f"[resume] WARN: could not parse epoch number from '{_resume_basename}'; starting from epoch 0"
                )

        # Resuming a state that's already at the final epoch trains nothing — the epoch loop is
        # empty and we fall through to writing the final LoRA. That fall-through is deliberate and
        # load-bearing (pausing ON the last epoch exits before the final LoRA is written, and
        # Resume is what writes it), so this must stay a loud message and NOT an error.
        if epoch_to_start >= num_train_epochs:
            accelerator.print(
                f"[resume] state is at epoch {epoch_to_start} of {num_train_epochs} — nothing left "
                f"to train. Writing the final LoRA from the restored state. To train further, "
                f"raise Max Train Epochs above {epoch_to_start} and resume again."
            )
            import sys as _sys; _sys.stdout.flush()

        # tqdm init AFTER resume parse so the progress bar shows the correct starting step count
        progress_bar = tqdm(
            range(args.max_train_steps), initial=global_step, smoothing=0,
            disable=not accelerator.is_local_main_process, desc="steps",
        )

        noise_scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")

        loss_recorder = LossRecorder()
        save_dtype = dit_dtype
        del train_dataset_group

        # ------------------------------------------------------------------
        # Save/remove helpers
        # ------------------------------------------------------------------

        def save_model(ckpt_name: str, unwrapped_nw, steps, epoch_no, force_sync_upload=False):
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)

            accelerator.print(f"\nSaving checkpoint: {ckpt_file}")
            metadata["ss_training_finished_at"] = str(time.time())
            metadata["ss_steps"] = str(steps)
            metadata["ss_epoch"] = str(epoch_no)

            metadata_to_save = minimum_metadata if args.no_metadata else metadata

            title = (args.metadata_title if args.metadata_title is not None
                     else resolve_title(args.output_name, args.metadata_trigger_phrase))
            if args.min_timestep is not None or args.max_timestep is not None:
                min_ts = args.min_timestep if args.min_timestep is not None else 0
                max_ts = args.max_timestep if args.max_timestep is not None else 1000
                md_timesteps = (min_ts, max_ts)
            else:
                md_timesteps = None

            thumb_arg = args.metadata_thumbnail
            if thumb_arg and thumb_arg.lower() in ("off", "none"):
                thumb_source = None
            elif thumb_arg:
                thumb_source = thumb_arg
            else:
                thumb_source = latest_sample_image(args.output_dir)

            sai_metadata = build_metadata(
                None,
                self.architecture,
                time.time(),
                title,
                args.metadata_reso,
                args.metadata_author,
                (args.metadata_description if args.metadata_description is not None
                 else getattr(self, "_last_sample_prompt", None)),
                args.metadata_license,
                args.metadata_tags,
                timesteps=md_timesteps,
                trigger_phrase=args.metadata_trigger_phrase,
                thumbnail=thumbnail_data_uri(thumb_source),
            )

            metadata_to_save.update(sai_metadata)
            unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)

        def remove_model(old_ckpt_name):
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                accelerator.print(f"Removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        # ------------------------------------------------------------------
        # Pre-training sample
        # ------------------------------------------------------------------

        if epoch_to_start > 0:
            accelerator.print(f"[resume] skipping initial sample (starting at epoch {epoch_to_start + 1})")
        elif should_sample_images(args, global_step, epoch=0):
            optimizer_eval_fn()
            # A preview failure must never end a training run (Krea 2 already degrades here).
            try:
                self.sample_images(accelerator, args, 0, global_step, vae, transformer, sample_parameters, dit_dtype)
            except Exception:
                logger.warning("sample generation failed — training continues, previews skipped "
                               "this round", exc_info=True)
            optimizer_train_fn()
        if len(accelerator.trackers) > 0:
            accelerator.log({}, step=0)

        # ------------------------------------------------------------------
        # Gradient Mining (experimental)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Training loop
        # ------------------------------------------------------------------

        unwrapped_transformer = accelerator.unwrap_model(transformer)
        first_param = next(iter(unwrapped_transformer.parameters()), None)
        logger.info(
            f"DiT dtype: {first_param.dtype if first_param is not None else None}, "
            f"device: {first_param.device if first_param is not None else accelerator.device}"
        )

        clean_memory_on_device(accelerator.device)
        optimizer_train_fn()

        # Adaptive LR state (bi-directional plateau tracker).
        # Hardcoded policy: start_epoch=1 (0-indexed), factor_up=1.25, factor_down=0.5,
        # patience=1 for epochs 1-3 until a stability event; patience=2 forever after
        # the first stability-triggered rollback (training is now in a delicate regime).
        if args.adaptive_lr:
            _initial_lr = optimizer.param_groups[0]["lr"]
            # Adafactor with relative_step manages its own LR and stores None in the param
            # groups — there is nothing for the watcher to adjust, and `None < float` was
            # a TypeError before step 1. Disable adaptive cleanly instead of crashing.
            if _initial_lr is None:
                accelerator.print(
                    "[adaptive_lr] DISABLED — the optimizer manages its own learning rate "
                    "(Adafactor relative_step stores lr=None), so there is no LR for the "
                    "watcher to adjust. Training continues without adaptive LR."
                )
                args.adaptive_lr = False
        if args.adaptive_lr:
            # The Learning Rate box is IGNORED while adaptive is on: the run starts at the
            # GEOMETRIC MIDPOINT of the Min/Max window and the watcher owns the LR from
            # there (probe up / ratchet down). Two knobs, not three — the old behaviour
            # (box = start, floor-clamped) silently made the box authoritative.
            # Resumed runs keep their restored LR: the watcher was mid-flight.
            if not args.resume:
                _mid = math.sqrt(args.adaptive_lr_min * args.adaptive_lr_max)
                if abs(_initial_lr - _mid) > 1e-12:
                    accelerator.print(
                        f"[adaptive_lr] starting LR set to {_mid:.3e} — the geometric midpoint "
                        f"of Min/Max (the Learning Rate box is ignored while adaptive is on)"
                    )
                for _g in optimizer.param_groups:
                    _g["lr"] = _mid
                _initial_lr = _mid
            accelerator.print(
                f"[adaptive_lr] ENABLED — starting_lr={_initial_lr:.3e} "
                f"min_lr={args.adaptive_lr_min:.3e} max_lr={args.adaptive_lr_max:.3e} "
                f"(watcher activates at end of epoch 1)"
            )
            import sys as _sys; _sys.stdout.flush()
        adaptive_best_loss = None
        adaptive_good_streak = 0
        adaptive_bad_streak = 0
        adaptive_stability_streak = 0          # consecutive red-signal epochs
        adaptive_stability_triggered = False   # pins patience=2 once set (loss AND stability)
        # Stability-signal counters (reset each epoch)
        adaptive_clip_events = 0
        adaptive_clip_steps = 0
        # Weight-norm tracking for epoch-over-epoch growth signal
        adaptive_prev_weight_norm = None
        # CPU snapshot of LoRA weights + optimizer state, taken at end of each epoch.
        # Used for 70/30 weight blend on stability-triggered reduce.
        # NOT preserved across pause/resume — too large for per-epoch JSON. First post-resume
        # epoch can't blend on a stability event; recaptured next epoch.
        adaptive_snapshot = None

        def _write_adaptive_sidecar(state_dir):
            """Drop adaptive_lr_state.json into a just-written state dir.

            Reads the watcher's scalars from the enclosing scope at call time, so it always
            records the CURRENT values. Called from the epoch boundary AND from the end-of-run
            save — miss the latter and a refinement pass resumed from a finished run restarts the
            watcher cold (best_loss=None, streaks 0), which the restore path accepts silently."""
            if not args.adaptive_lr:
                return
            try:
                import json as _json
                with open(os.path.join(state_dir, "adaptive_lr_state.json"), "w") as _f:
                    _json.dump({
                        "best_loss": adaptive_best_loss,
                        "good_streak": adaptive_good_streak,
                        "bad_streak": adaptive_bad_streak,
                        "stability_streak": adaptive_stability_streak,
                        "stability_triggered": adaptive_stability_triggered,
                        "prev_weight_norm": adaptive_prev_weight_norm,
                    }, _f, indent=2)
            except Exception as _e:
                accelerator.print(f"[adaptive_lr] sidecar save failed: {_e}")

        # Restore adaptive LR scalars from sidecar JSON on resume (small, ~6 scalars)
        if args.adaptive_lr and args.resume:
            import json as _json
            _adaptive_state_path = os.path.join(args.resume, "adaptive_lr_state.json")
            if os.path.exists(_adaptive_state_path):
                try:
                    with open(_adaptive_state_path, "r") as _f:
                        _ad = _json.load(_f)
                    adaptive_best_loss = _ad.get("best_loss")
                    adaptive_good_streak = int(_ad.get("good_streak", 0))
                    adaptive_bad_streak = int(_ad.get("bad_streak", 0))
                    adaptive_stability_streak = int(_ad.get("stability_streak", 0))
                    adaptive_stability_triggered = bool(_ad.get("stability_triggered", False))
                    adaptive_prev_weight_norm = _ad.get("prev_weight_norm")
                    accelerator.print(
                        f"[resume] adaptive_lr state restored: best_loss={adaptive_best_loss} "
                        f"streaks(g/b/s)={adaptive_good_streak}/{adaptive_bad_streak}/{adaptive_stability_streak} "
                        f"stability_triggered={adaptive_stability_triggered} "
                        f"prev_weight_norm={adaptive_prev_weight_norm}"
                    )
                    import sys as _sys; _sys.stdout.flush()
                except Exception as _e:
                    accelerator.print(f"[resume] WARN: failed to load adaptive_lr_state.json: {_e}")
        ADAPTIVE_BLEND_RATIO = 0.7  # 70% prev + 30% current on rollback
        # Stability thresholds (hardcoded)
        ADAPTIVE_CLIP_RATIO_THRESHOLD = 0.5   # >50% of steps clipping = too high
        ADAPTIVE_WEIGHT_GROWTH_THRESHOLD = 0.30  # >30% LoRA weight norm growth/epoch = too high

        # --- Perf diagnostic (opt-in: FIZGIG_PERF_DIAG=1) ---------------------
        # Pins the epoch-2-onward per-step slowdown to a cause. Per epoch logs:
        #   step_ms min/med  — min step = pure compute; watch it climb
        #   vram peak_alloc vs peak_reserved + gap + alloc retries
        #                    — reserved climbing while alloc flat (+ retries) = FRAGMENTATION
        #   hooks full_bwd   — climbing across epochs = block-swap backward-hook LEAK
        # Opt-in: set FIZGIG_PERF_DIAG=1 in the environment before launching to
        # enable. Off by default → zero overhead on normal runs.
        import os as _os, time as _time
        _perf_diag = _os.environ.get("FIZGIG_PERF_DIAG", "0") != "0"
        if _perf_diag:
            accelerator.print("[perfdiag] ACTIVE — a summary line prints at the end of each epoch.")
            import sys as _psys0; _psys0.stdout.flush()
        def _perf_hook_counts():
            mdl = accelerator.unwrap_model(transformer)
            f = b = fb = 0
            for mod in mdl.modules():
                f += len(getattr(mod, "_forward_hooks", {}) or {})
                b += len(getattr(mod, "_backward_hooks", {}) or {})
                fb += len(getattr(mod, "_full_backward_hooks", {}) or {})
            return f, b, fb

        for epoch in range(epoch_to_start, num_train_epochs):
            accelerator.print(f"\nepoch {epoch + 1}/{num_train_epochs}")
            current_epoch.value = epoch + 1
            metadata["ss_epoch"] = str(epoch + 1)

            accelerator.unwrap_model(network).on_epoch_start(transformer)

            _perf_step_times = []
            if _perf_diag and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            for step, batch in enumerate(train_dataloader):
                _perf_t0 = _time.perf_counter() if _perf_diag else 0.0
                # Warm-up reassurance: the first two epochs start slowly (first-sight kernel
                # planning, cuBLAS picks, allocator/cache warm-up) — a crawling bar looks
                # like a hang, so repeat a gentle note every ~30 s while it lasts.
                if epoch < 2:
                    _wu_now = _time.time()
                    if _wu_now - getattr(self, "_warmup_note_last", 0.0) > 30.0:
                        self._warmup_note_last = _wu_now
                        accelerator.print(
                            "[warm-up] Warm-up phase — the first two epochs start slowly while "
                            "the GPU plans kernels and fills its caches. Nothing is stuck; full "
                            "speed arrives from epoch 3."
                        )
                latents = batch["latents"]

                with accelerator.accumulate(training_model):
                    accelerator.unwrap_model(network).on_step_start()

                    # Klein: latents pass through without scaling
                    # (scale_shift_latents is identity for Klein)

                    # Sample noise
                    noise = torch.randn_like(latents)

                    # Get noisy input and timesteps
                    noisy_model_input, timesteps = self.get_noisy_model_input_and_timesteps(
                        args, noise, latents, batch["timesteps"], noise_scheduler, accelerator.device, dit_dtype
                    )

                    # Loss weighting
                    weighting = compute_loss_weighting_for_sd3(
                        args.weighting_scheme, noise_scheduler, timesteps, accelerator.device, dit_dtype
                    )

                    # Forward pass
                    model_pred, target = self.call_dit(
                        args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype
                    )

                    # MSE loss
                    loss = torch.nn.functional.mse_loss(model_pred.to(network_dtype), target, reduction="none")
                    if weighting is not None:
                        loss = loss * weighting
                    noise_loss = loss.mean()

                    loss = noise_loss

                    # Backward
                    accelerator.backward(loss)

                    # Gradient mining: directional filtering
                    if accelerator.sync_gradients:
                        # Manual gradient sync for DDP
                        state = accelerate.PartialState()
                        if state.distributed_type != accelerate.DistributedType.NO:
                            for param in network.parameters():
                                if param.grad is not None:
                                    param.grad = accelerator.reduce(param.grad, reduction="mean")

                        if args.max_grad_norm != 0.0:
                            params_to_clip = accelerator.unwrap_model(network).get_trainable_params()
                            pre_clip_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                            if args.adaptive_lr:
                                adaptive_clip_steps += 1
                                try:
                                    if float(pre_clip_norm) > float(args.max_grad_norm):
                                        adaptive_clip_events += 1
                                except (TypeError, ValueError):
                                    pass

                    optimizer.step()
                    if not args.adaptive_lr:
                        lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                # Weight norm regularization
                if args.scale_weight_norms:
                    keys_scaled, mean_norm, maximum_norm = accelerator.unwrap_model(network).apply_max_norm_regularization(
                        args.scale_weight_norms, accelerator.device
                    )
                    max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}
                else:
                    keys_scaled, mean_norm, maximum_norm = None, None, None

                # Progress tracking
                if accelerator.sync_gradients:
                    if global_step == 0:
                        progress_bar.reset()
                    progress_bar.update(1)
                    global_step += 1

                    should_sampling = should_sample_images(args, global_step, epoch=None)
                    should_saving = args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0

                    if should_sampling or should_saving:
                        optimizer_eval_fn()
                        if should_sampling:
                            self.sample_images(
                                accelerator, args, None, global_step, vae, transformer, sample_parameters, dit_dtype
                            )
                        if should_saving:
                            accelerator.wait_for_everyone()
                            if accelerator.is_main_process:
                                ckpt_name = get_step_ckpt_name(args.output_name, global_step)
                                save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch)

                                if args.save_state:
                                    save_and_remove_state_stepwise(args, accelerator, global_step)

                                remove_step_no = get_remove_step_no(args, global_step)
                                if remove_step_no is not None:
                                    remove_model(get_step_ckpt_name(args.output_name, remove_step_no))
                        optimizer_train_fn()

                current_loss = loss.detach().item()
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                avr_loss: float = loss_recorder.moving_average
                logs = {"avr_loss": avr_loss}
                progress_bar.set_postfix(**logs)

                if args.scale_weight_norms:
                    progress_bar.set_postfix(**{**max_mean_logs, **logs})

                if len(accelerator.trackers) > 0:
                    logs = self.generate_step_logs(
                        args, current_loss, avr_loss, lr_scheduler, lr_descriptions,
                        optimizer, keys_scaled, mean_norm, maximum_norm,
                    )
                    accelerator.log(logs, step=global_step)

                if _perf_diag:
                    _perf_step_times.append(_time.perf_counter() - _perf_t0)

                if global_step >= args.max_train_steps:
                    break

            # End of epoch
            if _perf_diag:
                import statistics as _pstats
                _st = sorted(_perf_step_times)
                _mn = (_st[0] * 1000) if _st else 0.0
                _md = (_pstats.median(_st) * 1000) if _st else 0.0
                _f, _b, _fb = _perf_hook_counts()
                if torch.cuda.is_available():
                    _pa = torch.cuda.max_memory_allocated() / 1e9
                    _pr = torch.cuda.max_memory_reserved() / 1e9
                    _ms = torch.cuda.memory_stats()
                    _rt = _ms.get("num_alloc_retries", 0)
                    _oo = _ms.get("num_ooms", 0)
                else:
                    _pa = _pr = _rt = _oo = 0
                accelerator.print(
                    f"[perfdiag] epoch {epoch + 1} | steps={len(_st)} | "
                    f"step_ms min={_mn:.1f} med={_md:.1f} | "
                    f"vram peak_alloc={_pa:.2f}G peak_reserved={_pr:.2f}G gap={_pr - _pa:.2f}G "
                    f"retries={_rt} ooms={_oo} | hooks fwd={_f} bwd={_b} full_bwd={_fb}"
                )
                import sys as _psys; _psys.stdout.flush()

            if len(accelerator.trackers) > 0:
                accelerator.log({"loss/epoch": loss_recorder.moving_average}, step=epoch + 1)

            if args.adaptive_lr and epoch == 0:
                _cur_lr = optimizer.param_groups[0]["lr"]
                adaptive_best_loss = loss_recorder.moving_average
                accelerator.print(
                    f"[adaptive_lr] epoch {epoch + 1:>2}: loss={loss_recorder.moving_average:.4f} "
                    f"lr={_cur_lr:.2e} | ARMED (baseline captured, watcher begins next epoch)"
                )
                import sys as _sys; _sys.stdout.flush()

            # Adaptive LR: bi-directional plateau tracker (epoch-boundary, loss + stability driven)
            if args.adaptive_lr and epoch >= 1:
                current_epoch_loss = loss_recorder.moving_average
                # Asymmetric patience:
                #   - probe UP: always 2 (the risky direction)
                #   - reduce DOWN ramp:
                #     * internal epoch 1 (first real comparison): 2 — first compare can be noisy
                #     * internal epochs 2-3: 1 — fast-reactive in early training
                #     * internal epoch 4+: 2 — conservative on the long tail
                #     * any time after a stability event: 2 — training has proven fragile
                patience_up = 2
                if adaptive_stability_triggered or epoch == 1 or epoch >= 4:
                    patience_down = 2
                else:
                    patience_down = 1

                # --- Collect signals ---
                clip_ratio = adaptive_clip_events / max(adaptive_clip_steps, 1)
                cur_weight_norm = 0.0
                with torch.no_grad():
                    for p in accelerator.unwrap_model(network).parameters():
                        if p.requires_grad:
                            cur_weight_norm += float(p.detach().float().norm().item()) ** 2
                cur_weight_norm = cur_weight_norm ** 0.5
                weight_growth = None
                if adaptive_prev_weight_norm is not None and adaptive_prev_weight_norm > 0:
                    weight_growth = (cur_weight_norm - adaptive_prev_weight_norm) / adaptive_prev_weight_norm

                stability_reason = None
                if clip_ratio > ADAPTIVE_CLIP_RATIO_THRESHOLD:
                    stability_reason = f"grad clip {clip_ratio*100:.0f}% > {ADAPTIVE_CLIP_RATIO_THRESHOLD*100:.0f}%"
                elif weight_growth is not None and weight_growth > ADAPTIVE_WEIGHT_GROWTH_THRESHOLD:
                    stability_reason = f"wnorm_Δ {weight_growth*100:+.0f}% > {ADAPTIVE_WEIGHT_GROWTH_THRESHOLD*100:.0f}%"

                # --- Decide action ---
                action = "HOLD"
                reason = ""
                cur_lr = optimizer.param_groups[0]["lr"]
                new_lr = cur_lr

                if stability_reason is not None:
                    adaptive_stability_streak += 1
                    # Require 1 red epoch before first fix, 2 consecutive red epochs thereafter
                    # (filters natural spikes and post-rollback residuals).
                    stability_patience = 1 if not adaptive_stability_triggered else 2
                    if adaptive_stability_streak >= stability_patience:
                        candidate = max(cur_lr * 0.5, args.adaptive_lr_min)
                        rollback_note = ""
                        if adaptive_snapshot is not None:
                            # Blend LoRA weights 70/30 toward prev-epoch snapshot, then fully restore optimizer state.
                            # Restoring optimizer state clears Adam's m/v that were pushing in the bad direction.
                            net_unwrapped = accelerator.unwrap_model(network)
                            current_params = dict(net_unwrapped.named_parameters())
                            with torch.no_grad():
                                for name, prev_tensor in adaptive_snapshot["weights"].items():
                                    if name in current_params and current_params[name].requires_grad:
                                        p = current_params[name]
                                        prev_on_device = prev_tensor.to(device=p.device, dtype=p.dtype)
                                        p.copy_(ADAPTIVE_BLEND_RATIO * prev_on_device + (1.0 - ADAPTIVE_BLEND_RATIO) * p)
                            try:
                                optimizer.load_state_dict(adaptive_snapshot["optim"])
                                rollback_note = f"; weights blended {int(ADAPTIVE_BLEND_RATIO*100)}/{int((1-ADAPTIVE_BLEND_RATIO)*100)}, optim restored"
                            except Exception as e:
                                rollback_note = f"; weights blended (optim restore failed: {e})"
                        if candidate < cur_lr:
                            new_lr = candidate
                            action = "REDUCE+ROLLBACK" if adaptive_snapshot is not None else "REDUCE"
                        else:
                            action = "HOLD (floored)"
                        reason = f"stability: {stability_reason}{rollback_note}"
                        adaptive_good_streak = 0
                        adaptive_bad_streak = 0
                        adaptive_stability_streak = 0
                        adaptive_stability_triggered = True  # pin patience at 2 from here on
                    else:
                        action = "WAIT"
                        reason = f"stability: {stability_reason}, streak {adaptive_stability_streak}/{stability_patience}"
                elif adaptive_best_loss is None or current_epoch_loss < adaptive_best_loss:
                    # Stable epoch — reset stability streak so "consecutive red" gating works
                    adaptive_stability_streak = 0
                    adaptive_best_loss = current_epoch_loss
                    adaptive_good_streak += 1
                    adaptive_bad_streak = 0
                    if adaptive_good_streak >= patience_up:
                        candidate = min(cur_lr * 1.25, args.adaptive_lr_max)
                        if candidate > cur_lr:
                            new_lr = candidate
                            action = "PROBE UP"
                            reason = f"loss improving, streak {adaptive_good_streak}"
                        else:
                            action = "HOLD (capped)"
                            reason = f"loss improving, streak {adaptive_good_streak}, at max_lr"
                        adaptive_good_streak = 0
                    else:
                        reason = f"loss improving, streak {adaptive_good_streak}/{patience_up}"
                else:
                    # Stable epoch (no stability signal, loss plateau) — reset stability streak
                    adaptive_stability_streak = 0
                    adaptive_bad_streak += 1
                    adaptive_good_streak = 0
                    if adaptive_bad_streak >= patience_down:
                        candidate = max(cur_lr * 0.5, args.adaptive_lr_min)
                        if candidate < cur_lr:
                            new_lr = candidate
                            action = "REDUCE"
                            reason = f"loss plateau, streak {adaptive_bad_streak}"
                        else:
                            action = "HOLD (floored)"
                            reason = f"loss plateau, streak {adaptive_bad_streak}, at min_lr"
                        adaptive_bad_streak = 0
                    else:
                        reason = f"loss plateau, streak {adaptive_bad_streak}/{patience_down}"

                # Apply LR change if any
                if new_lr != cur_lr:
                    for pg in optimizer.param_groups:
                        pg["lr"] = new_lr

                # One-line epoch summary (always printed when adaptive is on)
                lr_str = f"{cur_lr:.2e}" if new_lr == cur_lr else f"{cur_lr:.2e}→{new_lr:.2e}"
                wn_str = f"{weight_growth*100:+.0f}%" if weight_growth is not None else "—"
                accelerator.print(
                    f"[adaptive_lr] epoch {epoch + 1:>2}: loss={current_epoch_loss:.4f} "
                    f"lr={lr_str} clip={clip_ratio*100:.0f}% wnorm_Δ={wn_str} | {action} ({reason})"
                )
                import sys as _sys; _sys.stdout.flush()

                # Reset counters for next epoch; keep weight norm as baseline
                adaptive_prev_weight_norm = cur_weight_norm
                adaptive_clip_events = 0
                adaptive_clip_steps = 0

            # Snapshot LoRA weights + optimizer state on CPU for potential rollback next epoch.
            # Taken AFTER adaptive adjustments (so the snapshot reflects the state we ended up with).
            if args.adaptive_lr:
                with torch.no_grad():
                    net_unwrapped = accelerator.unwrap_model(network)
                    weight_snap = {
                        name: p.detach().clone().to("cpu")
                        for name, p in net_unwrapped.named_parameters()
                        if p.requires_grad
                    }
                try:
                    optim_sd = optimizer.state_dict()
                    # Clone tensors to CPU so snapshot survives future optimizer updates
                    def _to_cpu(obj):
                        if isinstance(obj, torch.Tensor):
                            return obj.detach().clone().to("cpu")
                        if isinstance(obj, dict):
                            return {k: _to_cpu(v) for k, v in obj.items()}
                        if isinstance(obj, list):
                            return [_to_cpu(v) for v in obj]
                        return obj
                    adaptive_snapshot = {"weights": weight_snap, "optim": _to_cpu(optim_sd)}
                except Exception as e:
                    accelerator.print(f"[adaptive_lr] snapshot failed at epoch {epoch + 1}: {e}")
                    adaptive_snapshot = {"weights": weight_snap, "optim": None}

            accelerator.wait_for_everyone()

            optimizer_eval_fn()
            # Pause flag check — if requested, force save this epoch even if save_every_n_epochs would skip
            pause_requested = bool(args.pause_flag_path and os.path.exists(args.pause_flag_path))
            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                if pause_requested:
                    saving = True  # force save on pause regardless of N
                if is_main_process and saving:
                    ckpt_name = get_epoch_ckpt_name(args.output_name, epoch + 1)
                    save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch + 1)

                    remove_epoch_no = get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch_no is not None:
                        remove_model(get_epoch_ckpt_name(args.output_name, remove_epoch_no))

                    if args.save_state or pause_requested:
                        _state_dir = save_state_on_epoch_end(args, accelerator, epoch + 1)
                        # Adaptive LR sidecar — restore-on-resume continuity for the watcher's scalars
                        _write_adaptive_sidecar(_state_dir)
                        # Prune LAST: the sidecar has to land in the dir we just wrote, and a
                        # prune that ran first could delete it out from under that write.
                        prune_state_dirs(args.output_dir, args.output_name, args.keep_last_n_states)

            # A preview failure must never end a run that might be hours in (Krea 2 already
            # degrades gracefully here) — the checkpoint for this epoch is already saved.
            try:
                self.sample_images(accelerator, args, epoch + 1, global_step, vae, transformer, sample_parameters, dit_dtype)
            except Exception:
                logger.warning("sample generation failed at epoch %d — training continues, "
                               "previews skipped this round", epoch + 1, exc_info=True)
            optimizer_train_fn()

            # Graceful pause — after save_state + sample_images for this epoch are complete, exit cleanly
            if pause_requested:
                accelerator.print(
                    f"[pause] requested via {args.pause_flag_path}. State saved at "
                    f"{args.output_name}-{epoch + 1:06d}-state. Exiting cleanly to free GPU memory."
                )
                # Consume the flag HERE: relying on the GUI to remove it left it armed
                # after a window close / crash, truncating the next run to one epoch.
                try:
                    os.remove(args.pause_flag_path)
                except OSError:
                    pass
                import sys as _sys; _sys.stdout.flush()
                _sys.exit(0)

        # ------------------------------------------------------------------
        # Training complete
        # ------------------------------------------------------------------

        metadata["ss_training_finished_at"] = str(time.time())

        if is_main_process:
            network = accelerator.unwrap_model(network)

        accelerator.end_training()
        optimizer_eval_fn()

        # End-of-run state. Skipped when the run trained nothing (resumed from a state already at
        # the final epoch): the only dir we'd write is the one we resumed FROM, and save_state
        # overwrites in place — a crash mid-rewrite would destroy the state for no gain.
        if is_main_process and args.save_state_on_train_end and num_train_epochs > epoch_to_start:
            _final_state = save_state_on_train_end(args, accelerator, num_train_epochs)
            _write_adaptive_sidecar(_final_state)
            prune_state_dirs(args.output_dir, args.output_name, args.keep_last_n_states)

        if is_main_process:
            ckpt_name = get_last_ckpt_name(args.output_name)
            save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=True)
            logger.info("Model saved.")

    # ------------------------------------------------------------------
    # Debug: show timestep distribution
    # ------------------------------------------------------------------

    def _show_timesteps(self, args: argparse.Namespace):
        """Visualize timestep sampling distribution (debug utility)."""
        N_TRY = 100000
        BATCH_SIZE = 1000
        CONSOLE_WIDTH = 64
        N_TIMESTEPS_PER_LINE = 25

        noise_scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")
        latents = torch.zeros(BATCH_SIZE, 128, 1024 // 16, 1024 // 16, dtype=torch.float16)
        noise = torch.ones_like(latents)

        sampled_timesteps = [0] * noise_scheduler.config.num_train_timesteps
        for _ in tqdm(range(N_TRY // BATCH_SIZE)):
            bucketed_timesteps = None
            if args.num_timestep_buckets is not None and args.num_timestep_buckets > 1:
                self.num_timestep_buckets = args.num_timestep_buckets
                bucketed_timesteps = [self.get_bucketed_timestep() for _ in range(BATCH_SIZE)]

            noisy, actual_timesteps = self.get_noisy_model_input_and_timesteps(
                args, noise, latents, bucketed_timesteps, noise_scheduler, "cpu", torch.float16
            )
            # For Klein 4D latents: (B, C, H, W)
            for t_val in actual_timesteps:
                t_int = int(t_val.item())
                if 0 <= t_int < len(sampled_timesteps):
                    sampled_timesteps[t_int] += 1

        sampled_timesteps_arr = np.array(sampled_timesteps)
        sampled_timesteps_arr = sampled_timesteps_arr.reshape(-1, N_TIMESTEPS_PER_LINE).mean(axis=1)

        max_count = max(sampled_timesteps_arr) if max(sampled_timesteps_arr) > 0 else 1
        print(f"Sampled timesteps: max count={max_count}")
        for i, t in enumerate(sampled_timesteps_arr):
            line = f"{i * N_TIMESTEPS_PER_LINE:4d}-{(i + 1) * N_TIMESTEPS_PER_LINE - 1:4d}: "
            line += "#" * int(t / max_count * CONSOLE_WIDTH)
            print(line)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _str_to_dtype(s: str) -> torch.dtype:
    """Convert a string to a torch dtype."""
    mapping = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn": torch.float8_e4m3fn,
    }
    s = s.lower().strip()
    if s in mapping:
        return mapping[s]
    raise ValueError(f"Unknown dtype: {s}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def setup_parser() -> argparse.ArgumentParser:
    """Build the complete argument parser for Klein 9B LoRA training."""

    def int_or_float(value):
        if value.endswith("%"):
            try:
                return float(value[:-1]) / 100.0
            except ValueError:
                raise argparse.ArgumentTypeError(f"Value '{value}' is not a valid percentage")
        try:
            float_value = float(value)
            if float_value >= 1 and float_value.is_integer():
                return int(value)
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{value}' is not an int or float")

    parser = argparse.ArgumentParser(description="Klein 9B LoRA Trainer")

    # ---- General ----
    parser.add_argument("--config_file", type=str, default=None, help="TOML config file for hyperparameters")
    parser.add_argument("--dataset_config", type=pathlib.Path, default=None, help="Dataset config file (TOML)")

    # ---- Attention ----
    parser.add_argument("--sdpa", action="store_true", help="Use scaled dot-product attention (PyTorch 2.0+)")
    parser.add_argument("--flash_attn", action="store_true", help="Use FlashAttention")
    parser.add_argument("--sage_attn", action="store_true", help="Use SageAttention (not supported for training)")
    parser.add_argument("--xformers", action="store_true", help="Use xformers for attention")
    parser.add_argument("--flash3", action="store_true", help="Use FlashAttention 3")
    parser.add_argument("--split_attn", action="store_true", help="Split attention (batch_size=1 per split)")
    parser.add_argument("--img_in_txt_in_offloading", action="store_true", help="Offload img_in/txt_in to CPU (no-op for Klein)")

    # ---- Torch compile ----
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (requires Triton)")
    parser.add_argument("--compile_backend", type=str, default="inductor", help="torch.compile backend")
    parser.add_argument("--compile_mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    parser.add_argument("--compile_dynamic", type=str, default=None, choices=["true", "false", "auto"])
    parser.add_argument("--compile_fullgraph", action="store_true")
    parser.add_argument("--compile_cache_size_limit", type=int, default=None)

    # ---- CUDA ----
    parser.add_argument("--cuda_allow_tf32", action="store_true", help="Allow TF32 on Ampere+ GPUs")
    parser.add_argument("--cuda_cudnn_benchmark", action="store_true", help="Enable cuDNN benchmark")

    # ---- Training ----
    parser.add_argument("--max_train_steps", type=int, default=1600, help="Total training steps")
    parser.add_argument("--max_train_epochs", type=int, default=None, help="Training epochs (overrides max_train_steps)")
    parser.add_argument("--max_data_loader_n_workers", type=int, default=8, help="Max DataLoader workers")
    parser.add_argument("--persistent_data_loader_workers", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--gradient_checkpointing_cpu_offload", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])

    # ---- Logging ----
    parser.add_argument("--logging_dir", type=str, default=None)
    parser.add_argument("--log_with", type=str, default=None, choices=["tensorboard", "wandb", "all"])
    parser.add_argument("--log_prefix", type=str, default=None)
    parser.add_argument("--log_tracker_name", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--log_tracker_config", type=str, default=None)
    parser.add_argument("--wandb_api_key", type=str, default=None)
    parser.add_argument("--log_config", action="store_true")

    # ---- DDP ----
    parser.add_argument("--ddp_timeout", type=int, default=None, help="DDP timeout in minutes")
    parser.add_argument("--ddp_gradient_as_bucket_view", action="store_true")
    parser.add_argument("--ddp_static_graph", action="store_true")

    # ---- Sampling ----
    parser.add_argument("--sample_every_n_steps", type=int, default=None)
    parser.add_argument("--sample_at_first", action="store_true")
    parser.add_argument("--sample_every_n_epochs", type=int, default=None)
    parser.add_argument("--sample_prompts", type=str, default=None, help="Prompt file for sample generation")
    parser.add_argument("--sample_dit", type=str, default=None,
                        help="Path to Distilled DiT for 4-step sample generation. "
                             "Loads separately alongside the training Base model.")
    parser.add_argument("--sample_blocks_to_swap", type=int, default=0,
                        help="Block swap for Distilled sampling DiT (from inference prefs).")
    parser.add_argument("--sample_int8", action="store_true",
                        help="INT8 (W8A8) fast matmul for the Distilled sample DiT (experimental, same VRAM as fp8).")
    parser.add_argument("--cache_sample_model", type=str, default="auto",
                        choices=["auto", "on", "off"],
                        help="Keep the Distilled sample model in CPU RAM between epochs "
                             "to skip the per-epoch disk reload. auto = only when free RAM "
                             "is comfortable; on = always; off = reload each time.")
    parser.add_argument("--sample_ref_image", type=str, default=None,
                        help="Reference image to edit-condition training samples on (Klein "
                             "is an edit model). Auto-capped to ~0.20 MP before encode so any "
                             "size is safe. The live sample override can swap it per-run.")

    # ---- Optimizer ----
    parser.add_argument("--optimizer_type", type=str, default="",
                        help="Optimizer: AdamW, AdamW8bit, or full.module.path.ClassName")
    parser.add_argument("--optimizer_args", type=str, default=None, nargs="*")
    parser.add_argument("--learning_rate", type=float, default=2.0e-6)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm (0 = no clipping)")

    # ---- LR Scheduler ----
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help="LR scheduler: linear, cosine, cosine_with_restarts, polynomial, constant, "
                             "constant_with_warmup, rex")
    parser.add_argument("--lr_warmup_steps", type=int_or_float, default=0)
    parser.add_argument("--lr_decay_steps", type=int_or_float, default=0)
    parser.add_argument("--lr_scheduler_num_cycles", type=int, default=1)
    parser.add_argument("--lr_scheduler_power", type=float, default=1)
    parser.add_argument("--lr_scheduler_timescale", type=int, default=None)
    parser.add_argument("--lr_scheduler_min_lr_ratio", type=float, default=None)
    parser.add_argument("--lr_scheduler_type", type=str, default="")
    parser.add_argument("--lr_scheduler_args", type=str, default=None, nargs="*")

    # ---- Adaptive LR (bi-directional plateau tracker) ----
    # When enabled, the main lr_scheduler.step() is skipped; LR is adjusted at epoch
    # boundaries based on loss_recorder.moving_average. Starts at epoch 1 (0-indexed).
    parser.add_argument("--adaptive_lr", action="store_true",
                        help="Enable bi-directional adaptive LR. Overrides step-based scheduler with a "
                             "loss-driven watcher that can probe UP (when loss descends steadily) or "
                             "reduce DOWN (when loss plateaus).")
    parser.add_argument("--adaptive_lr_min", type=float, default=1e-5,
                        help="Floor for adaptive LR reductions (below 1e-5 is effectively paused; use earlier-epoch checkpoints for recovery instead).")
    parser.add_argument("--adaptive_lr_max", type=float, default=4e-4,
                        help="Ceiling for adaptive LR increases (4e-4 is the empirical ceiling "
                             "for rank 4:4 Klein 9B LoRAs; the GUI always passes an explicit value).")

    # ---- Context LoRA (train new LoRA with existing one frozen + active) ----
    parser.add_argument("--context_lora_path", type=str, default=None,
                        help="Path to a .safetensors LoRA to apply (frozen) to the base model "
                             "during training. New LoRA learns to coexist with this context.")
    parser.add_argument("--context_lora_strength", type=float, default=1.0,
                        help="Strength multiplier for the context LoRA (typical: 0.5-1.0).")

    # ---- FP8 ----
    parser.add_argument("--fp8_base", action="store_true", help="Use fp8 for base model")
    parser.add_argument("--fp8_scaled", action="store_true", help="Use scaled fp8 for DiT")
    parser.add_argument("--quant_4bit", action="store_true",
                        help="Quantize the frozen base to 4-bit NF4 (low-VRAM mode, ~halves DiT residency). "
                             "Trains a LoRA on top, QLoRA-style. Incompatible with block swap (forced off).")

    # ---- Dynamo ----
    parser.add_argument("--dynamo_backend", type=str, default="NO", choices=[e.value for e in DynamoBackend])
    parser.add_argument("--dynamo_mode", type=str, default=None, choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--dynamo_fullgraph", action="store_true")
    parser.add_argument("--dynamo_dynamic", action="store_true")

    # ---- Block swap ----
    parser.add_argument("--blocks_to_swap", type=int, default=None)
    parser.add_argument("--use_pinned_memory_for_block_swap", action="store_true")
    parser.add_argument("--disable_numpy_memmap", action="store_true")

    # ---- Timestep sampling ----
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Embedded CFG guidance scale for training")
    parser.add_argument("--timestep_sampling", default="sigma",
                        choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift", "flux2_shift",
                                 "logsnr", "qinglong_flux"])
    parser.add_argument("--discrete_flow_shift", type=float, default=1.0)
    parser.add_argument("--sigmoid_scale", type=float, default=1.0)
    parser.add_argument("--weighting_scheme", type=str, default="none",
                        choices=["logit_normal", "mode", "cosmap", "sigma_sqrt", "none"])
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--min_timestep", type=int, default=None, help="Min timestep (0-999)")
    parser.add_argument("--max_timestep", type=int, default=None, help="Max timestep (1-1000)")
    parser.add_argument("--preserve_distribution_shape", action="store_true")
    parser.add_argument("--num_timestep_buckets", type=int, default=None)
    parser.add_argument("--show_timesteps", type=str, default=None, choices=["image", "console"])

    # ---- Network (LoRA) ----
    parser.add_argument("--no_metadata", action="store_true")
    parser.add_argument("--network_weights", type=str, default=None, help="Pretrained network weights")
    parser.add_argument("--network_module", type=str, default=None, help="Network module to train")
    parser.add_argument("--network_dim", type=int, default=None, help="Network rank/dimension")
    parser.add_argument("--network_alpha", type=float, default=1, help="LoRA alpha for weight scaling")
    parser.add_argument("--network_dropout", type=float, default=None)
    parser.add_argument("--network_args", type=str, default=None, nargs="*")
    parser.add_argument("--training_comment", type=str, default=None)
    parser.add_argument("--dim_from_weights", action="store_true")
    parser.add_argument("--scale_weight_norms", type=float, default=None)
    parser.add_argument("--base_weights", type=str, default=None, nargs="*")
    parser.add_argument("--base_weights_multiplier", type=float, default=None, nargs="*")

    # ---- Output ----
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for checkpoints")
    parser.add_argument("--output_name", type=str, default=None, help="Base name for output files")
    parser.add_argument("--resume", type=str, default=None, help="Path to training state to resume from")
    parser.add_argument("--save_every_n_epochs", type=int, default=None)
    parser.add_argument("--save_every_n_steps", type=int, default=None)
    parser.add_argument("--save_last_n_epochs", type=int, default=None)
    parser.add_argument("--save_last_n_steps", type=int, default=None)
    parser.add_argument("--save_last_n_steps_state", type=int, default=None)
    parser.add_argument("--save_state", action="store_true",
                        help="Write a resumable <name>-NNNNNN-state/ dir at every checkpoint.")
    parser.add_argument("--save_state_on_train_end", action="store_true",
                        help="Write a resumable state dir when the run finishes, so a completed "
                             "LoRA can be trained further by raising --max_train_epochs.")
    parser.add_argument("--keep_last_n_states", type=int, default=2,
                        help="Keep only the N newest state dirs (each is LoRA + optimizer, "
                             "hundreds of MB). Clamped to >= 1.")
    parser.add_argument("--pause_flag_path", type=str, default=None,
                        help="Sentinel file path. If present at end of an epoch, trainer "
                             "saves state (forced) and exits cleanly to free GPU memory.")

    # ---- Metadata ----
    parser.add_argument("--metadata_title", type=str, default=None)
    parser.add_argument("--metadata_author", type=str, default=None)
    parser.add_argument("--metadata_description", type=str, default=None)
    parser.add_argument("--metadata_license", type=str, default=None)
    parser.add_argument("--metadata_tags", type=str, default=None)
    parser.add_argument("--metadata_reso", type=str, default=None)
    parser.add_argument("--metadata_arch", type=str, default=None)
    parser.add_argument("--metadata_trigger_phrase", type=str, default=None,
                        help="Trigger word(s) recorded as modelspec.trigger_phrase")
    parser.add_argument("--metadata_thumbnail", type=str, default=None,
                        help="Path to an image to embed as modelspec.thumbnail. Omit to "
                             "auto-use the latest sample preview; pass 'off' to disable.")

    # ---- Model paths ----
    parser.add_argument("--dit", type=str, help="Path to DiT checkpoint (.safetensors)")
    parser.add_argument("--dit_dtype", type=str, default=None, help="DiT data type (default: bfloat16)")
    parser.add_argument("--vae", type=str, help="Path to VAE checkpoint (.safetensors)")
    parser.add_argument("--vae_dtype", type=str, default=None, help="VAE data type (default: float32)")

    # ---- Klein-specific ----
    parser.add_argument("--text_encoder", type=str, default=None, help="Path to Qwen3 text encoder checkpoint")
    parser.add_argument("--fp8_text_encoder", action="store_true", help="Use fp8 for Text Encoder")
    from fizgig.klein.model_utils import add_model_version_args
    add_model_version_args(parser)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = setup_parser()
    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    # Klein defaults
    if args.vae_dtype is None:
        args.vae_dtype = "float32"

    trainer = KleinTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
