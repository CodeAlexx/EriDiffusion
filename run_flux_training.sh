#!/bin/bash
# Quick script to run Flux LoRA training with proper settings

echo "Starting Flux LoRA Training"
echo "=========================="

# Ensure we're in the right directory
cd /home/alex/diffusers-rs/eridiffusion

# Set environment variables
export FLUX_LORA_ONLY=1
export RUST_BACKTRACE=1
export CUDA_VISIBLE_DEVICES=0

# Check if trainer is built
if [ ! -f "./target/release/trainer" ]; then
    echo "Building trainer..."
    cargo build --release --bin trainer
fi

# Run training
echo ""
echo "Running trainer with flux_lora_minimal.yaml"
echo "Make sure to update the dataset path in the config!"
echo ""

./target/release/trainer config/flux_lora_minimal.yaml