#!/bin/bash
# Generate real SDXL images using candle-transformers

echo "=========================================="
echo "Generating REAL SDXL 'white swan on mars'"
echo "=========================================="

# Set up paths
UNET_PATH="/home/alex/SwarmUI/Models/diffusion_models/sd_xl_base_1.0.safetensors"
VAE_PATH="/home/alex/SwarmUI/Models/VAE/sdxl_vae.safetensors"
OUTPUT_DIR="outputs/sdxl_lora/samples"

# Create output directory
mkdir -p $OUTPUT_DIR

# Check if model files exist
if [ ! -f "$UNET_PATH" ]; then
    echo "Error: SDXL model not found at $UNET_PATH"
    exit 1
fi

# Run SDXL generation using candle example
echo "Running SDXL inference..."
cd /home/alex/.cargo/registry/src/*/candle-examples-*/

cargo run --example stable-diffusion --release --features cuda \
    -- --prompt "a white swan on mars" \
       --sd-version xl \
       --height 1024 \
       --width 1024 \
       --n-steps 30 \
       --seed 42 \
       --num-samples 1 \
       --final-image "$OUTPUT_DIR/sdxl_real_swan_on_mars.jpg" \
       --unet-weights "$UNET_PATH" \
       --vae-weights "$VAE_PATH" 2>&1 | tee sdxl_generation.log

echo "✅ SDXL generation complete!"
echo "Output saved to: $OUTPUT_DIR/sdxl_real_swan_on_mars.jpg"