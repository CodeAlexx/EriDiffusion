#!/bin/bash

echo "=== Complete Fresh Start for Flux Training ==="

# 1. Clear ALL cached data
echo "Clearing all cached data..."
rm -rf /home/alex/diffusers-rs/datasets/*/.latent_cache
rm -rf /home/alex/diffusers-rs/datasets/*/.cache
rm -rf /tmp/flux_*
rm -rf ~/.cache/candle_flux*

# 2. Clear CUDA cache
echo "Clearing CUDA cache..."
rm -rf ~/.nv/ComputeCache/*

# 3. Set environment
echo "Setting environment..."
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_LAUNCH_BLOCKING=1

# 4. Run training
echo "Starting training with fresh cache..."
cd /home/alex/diffusers-rs
./eridiffusion/target/release/trainer config/train.yaml