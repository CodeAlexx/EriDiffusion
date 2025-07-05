#!/bin/bash

# Since our ai-toolkit-rs SD3.5 implementation is incomplete,
# let's use the working Candle SD3 example that already exists

echo "🎨 Generating SD 3.5 image using working Candle implementation..."

# Navigate to candle directory
cd /home/alex/diffusers-rs/candle-official

# Use the existing compiled binary
if [ ! -f "./target/release/examples/stable-diffusion-3" ]; then
    echo "Building Candle SD3 example..."
    cargo build --release --example stable-diffusion-3
fi

# Generate the image
echo "Generating 'a lady at the beach' with SD 3.5..."
./target/release/examples/stable-diffusion-3 \
    --which 3.5-large \
    --prompt "a lady at the beach" \
    --height 768 \
    --width 768 \
    --num-inference-steps 20 \
    --cfg-scale 4.0 \
    --seed 42

# Move to a better location
if [ -f "out.jpg" ]; then
    mv out.jpg lady_at_beach_sd35_768x768.jpg
    echo "✅ Image saved as: lady_at_beach_sd35_768x768.jpg"
else
    echo "❌ Generation failed"
fi