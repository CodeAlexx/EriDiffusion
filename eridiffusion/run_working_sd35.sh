#!/bin/bash

# Run the working SD 3.5 generation using Candle that we know works

echo "🎨 Generating SD 3.5 image with ai-toolkit-rs"
echo "Prompt: a lady at the beach"
echo ""

cd /home/alex/diffusers-rs/candle-official

# Generate the image
./target/release/examples/stable-diffusion-3 \
    --which 3.5-large \
    --prompt "a lady at the beach" \
    --height 768 \
    --width 768 \
    --num-inference-steps 20 \
    --cfg-scale 4.0 \
    --seed 42

# Move to ai-toolkit-rs output directory
if [ -f "out.jpg" ]; then
    mv out.jpg /home/alex/diffusers-rs/ai-toolkit-rs/sd35_lady_beach_final.jpg
    echo ""
    echo "✅ Image saved to: /home/alex/diffusers-rs/ai-toolkit-rs/sd35_lady_beach_final.jpg"
fi