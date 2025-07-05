#!/bin/bash
# Example script for training a LoRA adapter

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Train LoRA on custom dataset
ai-toolkit train \
  --config examples/train_config.yaml \
  --output-dir ./outputs/my_lora_$(date +%Y%m%d_%H%M%S)

# Convert to different formats if needed
ai-toolkit convert \
  --input ./outputs/my_lora_*/checkpoint-final \
  --output ./outputs/my_lora.safetensors \
  --to-format safetensors

echo "Training complete! LoRA saved to ./outputs/my_lora.safetensors"