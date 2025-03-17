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
    parser = argparse.ArgumentParser(description="Trainig script for a text-to-image generation example with GLIGEN model.")
    
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

    # Train
    parser.add_argument(
        "--input_perturbation",
        type=float,
        default=0,
        help="The scale of input perturbation. Recommended 0.1."
    )
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
    accelerator_proj_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
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
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    
    def deepspeed_zero3_init_disabled_ctx_manager():
        """
            Returns either a context list that includes one that will disable `zero.Init` or an empty context list.
        """
        
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerator.state.is_initialized() else None
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
        text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_or_path, subfolder="text_encoder", revision=args.revision)
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_or_path, subfolder="vae", revision=args.revision)
    
    
    