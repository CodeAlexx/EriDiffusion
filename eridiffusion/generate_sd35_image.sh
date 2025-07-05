#!/bin/bash

# SD 3.5 Image Generation Script
# Generates "a lady at the beach" using local SD 3.5 models

# Model paths
MMDIT_PATH="/home/alex/SwarmUI/Models/diffusion_models/sd3.5_large.safetensors"
VAE_PATH="/home/alex/SwarmUI/Models/VAE/OfficialStableDiffusion/sd35_vae.safetensors"
HF_CACHE="$HOME/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3.5-large"

# Check if models exist
if [ ! -f "$MMDIT_PATH" ]; then
    echo "Error: SD 3.5 model not found at $MMDIT_PATH"
    exit 1
fi

if [ ! -f "$VAE_PATH" ]; then
    echo "Error: SD 3.5 VAE not found at $VAE_PATH"
    exit 1
fi

# Navigate to sd35-lora-trainer directory (has working inference)
cd /home/alex/diffusers-rs/sd35-lora-trainer

# Build the generate binary if not already built
echo "Building SD 3.5 generator..."
cargo build --release --bin generate

# Create output directory
mkdir -p generated_images

# Run generation
echo "Generating image: 'a lady at the beach'"
./target/release/generate \
    --mmdit-checkpoint "$MMDIT_PATH" \
    --vae-checkpoint "$VAE_PATH" \
    --prompt "a lady at the beach" \
    --output "generated_images/lady_at_beach.png" \
    --width 1024 \
    --height 1024 \
    --steps 28 \
    --cfg-scale 7.0 \
    --seed 42 \
    --device cuda \
    --model-variant large

echo "Generation complete! Image saved as generated_images/lady_at_beach.png"
ls -la generated_images/