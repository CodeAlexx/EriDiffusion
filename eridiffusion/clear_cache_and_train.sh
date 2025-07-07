#!/bin/bash

# Script to clear cached latents and run training with consistent device IDs

echo "=== Clearing Cached Latents ==="
echo "This will remove cached tensors that have old DeviceIds from previous runs"

# Find and clear cache directories
CACHE_DIRS=$(find /home/alex/diffusers-rs/datasets -name ".latent_cache" -type d 2>/dev/null)

if [ -n "$CACHE_DIRS" ]; then
    echo "Found cache directories:"
    echo "$CACHE_DIRS"
    echo
    echo "Clearing caches..."
    for cache_dir in $CACHE_DIRS; do
        echo "Removing: $cache_dir"
        rm -rf "$cache_dir"
    done
    echo "Cache cleared!"
else
    echo "No cache directories found"
fi

echo
echo "=== Starting Training with Single Device ==="
echo "Using CUDA_VISIBLE_DEVICES=0 to ensure single GPU"
echo

# Set environment and run training
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Change to the eridiffusion directory
cd /home/alex/diffusers-rs/eridiffusion

# Run the trainer
if [ -f "./target/release/trainer" ]; then
    echo "Running release build..."
    ./target/release/trainer "$@"
elif [ -f "./target/debug/trainer" ]; then
    echo "Running debug build..."
    ./target/debug/trainer "$@"
else
    echo "Trainer not found! Building..."
    cargo build --release
    ./target/release/trainer "$@"
fi