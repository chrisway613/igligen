#!/bin/bash -x

export CUDA_VISIBLE_DEVICES=$(
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | \
    awk 'BEGIN {max=0; idx=-1} 
         {if ($1>max) {max=$1; idx=NR-1}} 
         END {print idx}'
)

# Model Name
MODEL_NAME="stabilityai/stable-diffusion-2-1-base"

# Training Setting
BATCH_SIZE_SINGLE_GPU=8
NUM_WORKERS=8
DATA_CONFIG_PATH="dataset/sam_full_boxtext2img.yaml"
EXP_NAME=gligen_sdv2.1_bs32

# Run scripts
accelerate launch --mixed_precision="fp16" \
  --main_process_port 0 train_text_to_image_gligen_sam.py \
  --pretrained_model_name_or_path=${MODEL_NAME} \
  --config=${DATA_CONFIG_PATH} \
  --resolution=512 \
  --train_batch_size=${BATCH_SIZE_SINGLE_GPU} \
  --gradient_accumulation_steps=2 --mixed_precision="fp16" \
  --max_train_steps=500000 \
  --learning_rate=5.e-05 --adam_weight_decay=0.0 --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=1000 \
  --output_dir=${EXP_NAME} --report_to=wandb \
  --dataloader_num_workers=${NUM_WORKERS} \
  --validation_steps=500 \
  --enable_flash_attention \
  --checkpointing_steps=1000 \
  --checkpoints_total_limit=2 \
  --prob_use_caption=0.5 --prob_use_boxes=0.9 \
  --no_caption_only \