#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and


import os
import shutil
import logging
import argparse

import numpy as np

import torch
import torch.utils.data
import torch.utils.checkpoint
import torch.nn.functional as F

import diffusers

import accelerate
import transformers

from pathlib import Path
from tqdm.auto import tqdm
from packaging import version

from huggingface_hub import create_repo, upload_folder

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed

from transformers.utils import ContextManagers
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionGLIGENPipeline, UNet2DConditionModel

from utils.parser import load_args
from dataset.sam_dataset import SAMDataset


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed.
# Remove at your own risks.
check_min_version("0.20.0")

logger = get_logger(__name__, log_level='INFO')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trainig script for a text-to-image generation example with GLIGEN model.")

    def box_type(values):
        try:
            box = list(map(float, values.split(',')))
            if len(box) != 4:
                raise argparse.ArgumentTypeError(
                    f"each box must have exactly 4 elements, got: {len(box)}")

            return box
        except:
            raise argparse.ArgumentTypeError(
                f"Invalid box format: {values}. Expect 4 comma-separated floating numbers.")

    # GLIGEN
    parser.add_argument(
        "--no_caption_only",
        action="store_true",
        help="whether to drop the caption when the boxes are dropped",
    )
    parser.add_argument("--prob_use_caption", type=float,
                        default=0.5, help="The prob of keeping caption.")
    parser.add_argument("--prob_use_boxes", type=float,
                        default=0.9, help="The prob of keeping boxes.")

    # Pretrained model
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--override_unet_weights",
        type=str,
        default=None,
        help="The path to weights to override the pretrained weights in UNet.",
    )

    # Dataset
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="Data path.",
    )
    parser.add_argument(
        "--config",
        metavar='C',
        type=str,
        nargs='?',
        default="./dataset/sam_boxtext2img.yaml",
        help="Path to data configuration file."
    )
    # From detectron2
    parser.add_argument(
        "--opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line 'KEY VALUE' pairs",
    )

    # Inference
    # Resolution is used in inference (we read latents directly in training)
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="An image of grassland with a dog.",
        help="text prompt for validation."
    )
    parser.add_argument(
        "--validation_ground_phrases",
        nargs='*',
        type=str,
        default=["a dog"],
        help="List of ground phrases for validation."
    )
    parser.add_argument(
        "--validation_ground_boxes",
        nargs='*',
        type=box_type,
        default=[[0.1, 0.6, 0.3, 0.8]],
        help="List of ground boxes for validation, boxes are separated by whitespace, "
             "and each box is a list of 4 floating numbers separated by comma: x1,y1,x2,y2."
    )

    # Train
    parser.add_argument(
        "--input_perturbation",
        type=float,
        default=0,
        help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument("--noise_offset", type=float,
                        default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
             "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. "
             "If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader."
    )
    # `num_train_epochs` is not supported (since we use iterable as dataset)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--no_semantic_grounding",
        action="store_true",
        help="If enabled, do not pass semantic information into grounding.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float,
                        default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08,
                        help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0,
                        type=float, help="Max gradient norm.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true",
                        help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=10,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--enable_flash_attention",
                        action="store_true", help="Enable flash attention.")

    # Outputs and records
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument("--hub_token", type=str, default=None,
                        help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="gligen-text2image-generation",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    # Global
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--seed", type=int, default=None,
                        help="A seed for reproducible training.")

    args = parser.parse_args()

    env_local_rank = os.getenv("LOCAL_RANK", -1)
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = int(env_local_rank)

    # Default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    print(f"Loading config from {args.config}..")
    config = load_args(args.config, cli_opts=args.opts)
    print(f"Config: {config}")

    return args, config


def main():
    args, config = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )

    # Accelerator
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_proj_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        project_config=accelerator_proj_config,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to
    )

    # Set logging format.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%D/%Y %H:%M:%S",
        level=logging.INFO
    )
    # Make one log on every process with the configuration for debugging.
    logger.info(accelerator.state, main_process_only=False)

    # Set logging level for different process.
    if accelerator.is_local_main_process:
        # datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbositu_error()
        diffusers.utils.logging.set_versbotisy_error()

    # Set random seed if provided.
    if args.seed is not None:
        set_seed(args.seed)

    # Enable Flash Attention if available.
    if args.enable_flash_attention:
        from flash_attn import flash_attn_func

        torch_sdp_attn = F.scaled_dot_product_attention

        def sdp_attn(q, k, v, attn_mask=None, dropout_p=0., is_causal=False):
            # torch convention: B, num heads, seq len, C
            assert attn_mask is None, f"attn_mask is not supported for Flash Attention."

            if q.size(-1) > 256:
                return torch_sdp_attn(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

            q, k, v = map(lambda x: x.permute(0, 2, 1, 3), (q, k, v))
            return flash_attn_func(q, k, v, dropout_p=dropout_p, is_causal=is_causal).permute(0, 2, 1, 3)

        F.scaled_dot_product_attention = sdp_attn
        logger.info(f"Flash Attention enabled.")

    # Use memory efficient attention as a fallback.
    torch.backends.cuda.enable_flash_sdp = False
    torch.backends.cuda.enable_mem_efficient_sdp = False
    torch.backends.cuda.enable_math_sdp = False

    # Handle the repository creation.
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token
            ).repo_id

    # Logging noise scheduler, tokenizer and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)

    def deepspeed_zero3_init_disabled_ctx_manager():
        """
            Returns either a context list that includes one that will disable `zero.Init` or an empty context list.
        """

        deepspeed_plugin = AcceleratorState(
        ).deepspeed_plugin if accelerator.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`.
    # But accelerate will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.

    # For now the following workaround will partially support Deepspeed ZeRO-3,
    # by excluding the 2 frozen models from being partitioned during `zero.Init`
    # which gets called during `from_pretrained`. So `CLIPTextModel` and `AutoencoderKL` will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.

    with ContextManagers(deepspeed_zero3_init_disabled_ctx_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_or_path, subfolder="text_encoder", revision=args.revision)
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_or_path, subfolder="vae", revision=args.revision)

    unet_gligen_kwargs = dict(
        attention_type="gated",
        low_cpu_mem_usage=False,
        device_map=None
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.non_ema_revision,
        **unet_gligen_kwargs
    )

    if args.override_unet_weights:
        logger.info(f"Override UNet weights with {args.override_unet_weights}")

        mismatches = unet.load_state_dict(
            torch.load(args.override_unet_weights, map_location="cpu"),
            strict=False
        )
        assert not len(
            mismatches.unexpected_keys), f"There are unexpected keys: {mismatches.unexpected_keys}"
        assert all([
            'fuser' in k or 'positon_net' in k
            for k in mismatches.missing_keys
        ]), f"There are missing keys that are not `fuser` or `position_net`: " \
            f"{[k for k in mismatches.missing_keys if not ('fuser' in k or 'position_net' in k)]}"

    logger.info(f"{unet}\n")

    # Freeze VAE and CLIPTextModel
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Make only parts of GLIGEN trainable.
    for name, param in unet.named_parameters():
        if '.fuser' in name or 'position_net' in name:
            param.requires_grad_(True)
            logger.info(f"Has grad param: {name}")
        else:
            param.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.revision,
            **unet_gligen_kwargs
        )
        ema_unet = EMAModel(ema_unet.parameters(
        ), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    # Enable xFormers if available.
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, "
                    "please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )

            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xFormers is not available, make sure it is installed correctlly.")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Customized saving, `accelerate` 0.16.0 will have better support for this.
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # Create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format.
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "ema_unet"))

            for model in models:
                model.save_pretrained(os.path.join(output_dir, "unet"))
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(
                    input_dir, "ema_unet"), model_cls=UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)

                del load_model

                for _ in range(len(models)):
                    model = models.pop()

                    load_model = UNet2DConditionModel.from_pretrained(
                        input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())

                    del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul_allow_tf32 = True

    # Set learning rate, optimizer and lr scheduler.
    if args.scale_lr:
        args.learning_rate *= accelerator.gradient_accumulation_steps * \
            args.train_batch_size * accelerator.num_processes

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes
    )

    # Dataset

    def transform(example):
        example['input_ids'] = tokenizer(
            example["caption"],
            padding="max_length",
            max_length=tokenizer.model_max_legnth,
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]

        return example

    # NOTE: `SAMDataset` repeats infinitely
    train_dataset = SAMDataset(
        data_path=args.data_path,
        train_shards=config.train_shards,
        prob_use_caption=args.prob_use_caption,
        prob_use_boxes=args.prob_use_boxes,
        box_confidence_th=0.25,
        batch_size=args.train_batch_size,
        transform=transform,
        shard_shuffle_seed=None,
        ddp_rank=accelerator.process_index,
        num_ddp_processes=accelerator.num_processes,
        no_caption_only=args.no_caption_only
    )

    # Dataloader

    def collate_fn(examples):
        # Batch size is set to 1 (we handle batching in the dataset)
        examples = examples[0]
        results = {}

        for k in ('latents', 'input_ids', 'boxes', 'text_masks'):
            results[k] = torch.stack([example[k] for example in examples])

            if k != 'input_ids':
                results[k] = results[k].float()
            if k == 'latents':
                results[k] = results[k].to(
                    memory_format=torch.contiguous_format)

        for k in ('id', 'box_phrases', 'caption'):
            results[k] = [example[k] for example in examples]

        return results

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        # iterable dataset does not support shuffling (shuffling is implemented in the dataset)
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        pin_memory=True
    )

    # We manually manage the train_dataloader (otherwise it will use `IterableDatasetShard`, which we do not need)
    unet, optimizer, lr_scheduler = accelerator.prepare(
        unet, optimizer, lr_scheduler)

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    mixed_precision = accelerator.mixed_precision
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = mixed_precision
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = mixed_precision

    vae.to(device=accelerator.device, dtype=weight_dtype)
    text_encoder(device=accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(
            args.tracker_project_name,
            config=tracker_config,
            init_kwargs={
                "wandb": {
                    "name": os.path.basename(args.output_dir)
                }
            }
        )
        # Log code with wandb
        wandb.run.log_code('.')

    # Star Training!
    total_batch_size = args.train_batch_size * \
        args.gradient_accumulatoin_steps * accelerator.num_processes

    logger.info("***** Running training *****")
    logger.info(
        f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda name: int(name.split('-')[1]))
            path = dirs[-1] if dirs else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist, starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(
                f"Resuming from checkpoint {args.resume_from_checkpoint}..")
            accelerator.load_state(os.path.join(args.output_dir, path))
            accelerator.print("Done!\n")

            global_step = int(path.split('-')[1])
            resume_step = global_step * args.gradient_accumulation_steps

    # Only show the progress bar once on each machine.
    # We skip steps so we should always from step 0.
    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    def compute_snr(timesteps):
        """
        Computes SNR as per 
        https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """

        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod ** 0.5
        sqrt_one_minus_alphas_cumprod = (1. - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(timesteps.device)[
            timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(timesteps.device)[
            timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # SNR
        return (alpha / sigma) ** 2

    # The dataset is repeating infinitely(so we only use 1 epoch).
    args.num_train_epochs = 1

    # Train loop
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update()

                continue

            # Preprocessing grounding text.
            ground_text_embed_batch = []
            for phrases_per_img in batch['box_phrases']:
                embed_per_img = torch.zeros(
                    (train_dataset.max_boxes_per_image,
                     unet.config.cross_attention_dim),
                    dtype=weight_dtype, device=accelerator.device
                )

                num_objs = len(phrases_per_img)
                if num_objs:
                    # Prepare batched input to the PositionNet (boxes, phrases, mask)
                    # Get tokens for phrases from pre-trained CLIPTokenizer
                    tokens = tokenizer(phrases_per_img, padding=True, return_tensors="pt").to(
                        accelerator.device)
                    embed_per_img[:num_objs] = text_encoder(
                        **tokens).pooler_output

                ground_text_embed_batch.append(embed_per_img)

            ground_text_embed_batch = torch.stack(ground_text_embed_batch)

            with accelerator.accumulate(unet):
                latents = batch['latents'].to(accelerator.device)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * \
                        torch.randn((noise.size(0), 1, 1, 1),
                                    device=noise.device)

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.size(0),),
                    device=latents.device, dtype=torch.long
                )

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    input_noise = noise + args.input_perturbation * \
                        torch.randn_like(noise)
                else:
                    input_noise = noise

                noisy_latents = noise_scheduler.add_noise(
                    latents, input_noise, timesteps)
                del input_noise

                # Get the text embedding for conditioning,
                # this is `last_hidden_states` from CLIPTextEncoder.
                # NOTE: we drop the caption text condition inside dataset.
                encoder_hidden_states = text_encoder(
                    batch['input_ids'].to(accelerator.device))[0]

                # Get the target for loss depending on the prediction type.
                if args.prediction_type is not None:
                    # Set prediction_type of scheduler if defined.
                    noise_scheduler.register_to_config(
                        prediction_type=args.prediction_type)

                pred_type = noise_scheduler.config.prediction_type
                if pred_type == 'epsilon':
                    target = noise
                elif pred_type == 'v_prediction':
                    target = noise_scheduler.get_velocity(
                        latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown precition type: {pred_type}")

                # Predict the noise residual and compute loss
                cross_attn_kwargs_for_gligen = {
                    'gligen': {
                        'positive_embeddings': ground_text_embed_batch,
                        'boxes': batch['boxes'].to(dtype=weight_dtype, device=accelerator.device),
                        'masks': batch['text_masks'].to(dtype=weight_dtype, device=accelerator.device)
                    }
                }
                pred = unet(noisy_latents, timesteps, encoder_hidden_states,
                            cross_attention_kwargs=cross_attn_kwargs_for_gligen).sample

                if args.snr_gamma is None:
                    loss = F.mse_loss(
                        pred.float(), target.float(), reduction='mean')
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(timesteps)
                    loss_weight = torch.stack(
                        [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr

                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    loss = F.mse_loss(
                        pred.float(), target.float(), reduction='none')
                    loss = loss_weight * \
                        loss.mean(dim=list(range(1, len(loss.shape))))
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(
                    args.train_batch_size)).mean().item()
                train_loss += avg_loss / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients():
                    accelerator.clip_grad_norm_(
                        unet.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())

                global_step += 1
                progress_bar.upadte()

                accelerator.log(
                    {'train_loss': train_loss},
                    step=global_step
                )
                train_loss = 0.

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = [d for d in os.listdir(
                                args.output_dir) if d.startswith('checkpoint')]
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(
                                    checkpoints) - args.checkpoints_total_limit + 1
                                checkpoints = sorted(
                                    checkpoints, key=lambda name: int(name.split('-')[1]))
                                removing_checkpoints = checkpoints[:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {num_to_remove} checkpoints\n"
                                    f"Removing checkpoints: {', '.join(removing_checkpoints)}\n"
                                )

                                for remove_ckpt in removing_checkpoints:
                                    remove_ckpt = os.path.join(
                                        args.output_dir, remove_ckpt)
                                    shutil.rmtree(remove_ckpt)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Save state to {save_path}\n")

                # Validation
                if global_step % args.validation_step == 0 and accelerator.is_main_process:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    if args.use_ema:
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())

                    log_validation(
                        args.validation_prompt,
                        args.validation_ground_phrases,
                        args.validation_ground_boxes,
                        text_encoder,
                        tokenizer,
                        vae,
                        unet,
                        args,
                        accelerator,
                        weight_dtype,
                        global_step,
                    )

                    if args.use_ema:
                        # Switch back to the original UNet parameters.
                        ema_unet.restore(unet.parameters())

            logs = {
                'step_loss': loss.detach().item(),
                'lr': lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionGLIGENPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=args.revision
        )
        pipeline.save_pretrained(args.output_dir)
        del pipeline

        if args.push_to_hub:
            # save_model_card(args, repo_id, images, repo_folder=args.output_dir)
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=['steps_', 'epoch_*']
            )

    accelerator.end_training()


def log_validation(
    validation_prompt,
    validation_ground_phrases,
    validation_ground_boxes,
    text_encoder, tokenizer,
    vae, unet, args, accelerator,
    weight_dtype, epoch,
    num_images_per_prompt=4,
    num_inference_steps=50
):
    logger.info("Running validation..\n")

    pipeline = StableDiffusionGLIGENPipeline.from_pretrained(
        vae=vae, text_encoder=text_encoder,
        tokenizer=tokenizer, unet=unet, safety_checker=None,
        revision=args.revision, torch_dtype=weight_dtype
    )
    pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_effficient_attention()

    generator = None
    if args.seed is not None:
        generator = torch.Generator(
            device=accelerator.device).manual_seed(args.seed)

    images = []
    for _ in range(num_images_per_prompt):
        with torch.autocast("cuda"):
            image = pipeline(
                validation_prompt, num_inference_steps=num_inference_steps,
                generator=generator, gligen_phrases=validation_ground_phrases,
                gligen_boxes=validation_ground_boxes, gligen_scheduled_sampling_beta=1.,
                height=args.resolution, width=args.resolution
            ).images[0]

        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(
                "validation", images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    'validation': [
                        wandb.Image(img, caption=f"{i}: {validation_prompt}")
                        for i, img in enumerate(images, start=1)
                    ]
                }
            )
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline, images
    torch.cuda.empty_cache()
