#!/bin/bash

# SD 3.5 Generation Script using local models

echo "SD 3.5 Image Generation with Local Models"
echo "========================================="

# Check if sd35-lora-trainer exists
if [ ! -d "../sd35-lora-trainer" ]; then
    echo "Error: sd35-lora-trainer not found!"
    echo "Expected at: ../sd35-lora-trainer"
    exit 1
fi

cd ../sd35-lora-trainer

# Build the project if needed
echo "Building SD 3.5 trainer..."
cargo build --release --bin generate 2>/dev/null || {
    echo "Build failed! Trying without CUDA..."
    cargo build --release --bin generate --no-default-features
}

# Model paths
MODELS_DIR="/home/alex/SwarmUI/Models"
HF_CACHE="/home/alex/.cache/huggingface/hub"
TEXT_ENCODERS="${HF_CACHE}/models--stabilityai--stable-diffusion-3.5-large/snapshots/764a7c2c5b58de7099a102985f91ca87b656c279/text_encoders"

# Check if models exist
echo -e "\nChecking for required models..."

if [ -f "${MODELS_DIR}/diffusion_models/sd3.5_large.safetensors" ]; then
    echo "✓ Found SD 3.5 Large model"
    MMDIT_PATH="${MODELS_DIR}/diffusion_models/sd3.5_large.safetensors"
else
    echo "✗ SD 3.5 Large model not found!"
    exit 1
fi

if [ -f "${MODELS_DIR}/VAE/OfficialStableDiffusion/sd35_vae.safetensors" ]; then
    echo "✓ Found SD 3.5 VAE"
    VAE_PATH="${MODELS_DIR}/VAE/OfficialStableDiffusion/sd35_vae.safetensors"
else
    echo "✗ SD 3.5 VAE not found!"
    exit 1
fi

# Generate image using the binary with local paths
echo -e "\nGenerating image..."
echo "Prompt: A beautiful mountain landscape at sunset, ultra detailed, masterpiece"

# Try with explicit model paths first
cargo run --release --bin generate -- \
    --prompt "A beautiful mountain landscape at sunset, ultra detailed, masterpiece" \
    --output "../ai-toolkit-rs/sd35_generated.png" \
    --model-variant large \
    --mmdit-checkpoint "${MMDIT_PATH}" \
    --vae-checkpoint "${VAE_PATH}" \
    --width 1024 \
    --height 1024 \
    --steps 28 \
    --cfg-scale 7.5 \
    --seed 42 \
    --device cuda 2>/dev/null || {
    
    echo "Failed with local paths, trying default HuggingFace model..."
    
    # Fallback to using HuggingFace model ID
    cargo run --release --bin generate -- \
        --prompt "A beautiful mountain landscape at sunset, ultra detailed, masterpiece" \
        --output "../ai-toolkit-rs/sd35_generated.png" \
        --model-variant large \
        --width 1024 \
        --height 1024 \
        --steps 28 \
        --cfg-scale 7.5 \
        --seed 42 \
        --device cuda
}

# Check if image was generated
if [ -f "../ai-toolkit-rs/sd35_generated.png" ]; then
    echo -e "\n✅ Success! Image generated at: ai-toolkit-rs/sd35_generated.png"
    echo "File info:"
    ls -lh ../ai-toolkit-rs/sd35_generated.png
else
    echo -e "\n❌ Failed to generate image"
    echo "Check the error messages above for details"
fi

# Return to original directory
cd ../ai-toolkit-rs