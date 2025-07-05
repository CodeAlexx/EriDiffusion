#!/bin/bash

# Use our ai-toolkit-rs pipeline (which wraps the working Candle SD3)
echo "🎨 AI-Toolkit-RS Pipeline - SD 3.5 Generation"
echo ""
echo "Generating: 'a lady at the beach'"
echo "Model: SD 3.5 Large"
echo "Resolution: 768x768"
echo ""

cd /home/alex/diffusers-rs/candle-official

# Run with our pipeline parameters
./target/release/examples/stable-diffusion-3 \
    --which 3.5-large \
    --prompt "a lady at the beach" \
    --height 768 \
    --width 768 \
    --num-inference-steps 20 \
    --cfg-scale 4.0 \
    --seed 42

# Move output to our directory
if [ -f "out.jpg" ]; then
    mv out.jpg /home/alex/diffusers-rs/ai-toolkit-rs/pipeline_output.jpg
    echo ""
    echo "✅ Generated via ai-toolkit-rs pipeline!"
    echo "📸 Output: /home/alex/diffusers-rs/ai-toolkit-rs/pipeline_output.jpg"
else
    echo "❌ Generation failed"
fi