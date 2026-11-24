#!/bin/bash
# LoRA v2 training (same settings as Rishit's v2 run, with the two paths fixed).
# Run from the repo root:  bash scripts/train.sh
accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --dynamo_backend="no" \
  --mixed_precision="fp16" \
  scripts/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --train_data_dir="data/processed_dataset" \
  --caption_column="text" \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --rank=4 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1500 \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=100 \
  --checkpointing_steps=500 \
  --report_to="tensorboard" \
  --output_dir="runs/lora_v2" \
  --mixed_precision="fp16" \
  --seed=42
