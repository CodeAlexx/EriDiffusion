#!/bin/bash

# SD 3.5 Generation using Candle's official example

echo "SD 3.5 Generation with Candle"
echo "============================="

# Navigate to candle examples
cd ../candle-official/candle-examples

# Build the SD3 example
echo "Building Candle SD3 example..."
cargo build --release --example stable-diffusion-3 --features cuda 2>/dev/null || {
    echo "CUDA build failed, trying CPU build..."
    cargo build --release --example stable-diffusion-3
}

# Run generation with SD 3.5 Large
echo -e "\nGenerating image with SD 3.5 Large..."
cargo run --release --example stable-diffusion-3 -- \
    --which 3.5-large \
    --prompt "A majestic dragon flying over a crystal castle, fantasy art, highly detailed" \
    --height 1024 \
    --width 1024 \
    --num-inference-steps 28 \
    --cfg-scale 7.5 \
    --output ../../ai-toolkit-rs/sd35_candle_output.png

# Check if successful
if [ -f "out.jpg" ]; then
    echo -e "\n✅ Image generated successfully!"
    # Move to ai-toolkit-rs directory
    mv out.jpg ../../ai-toolkit-rs/sd35_candle_output.jpg
    echo "Image saved to: ai-toolkit-rs/sd35_candle_output.jpg"
    cd ../../ai-toolkit-rs
    ls -lh sd35_candle_output.jpg
else
    echo -e "\n❌ Image generation failed"
fi

cd ../../ai-toolkit-rs