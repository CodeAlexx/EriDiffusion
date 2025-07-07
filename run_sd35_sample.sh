#!/bin/bash
# Simple script to run SD3.5 sampling using candle

PROMPT="$1"
OUTPUT="$2"
SEED="${3:-42}"

# Run candle's SD3 example with CUDA
cd /home/alex/diffusers-rs/candle-official/candle-examples
cargo run --release --features cuda --example stable-diffusion-3 -- \
    --prompt "$PROMPT" \
    --which 3.5-large \
    --height 1024 \
    --width 1024 \
    --num-inference-steps 25 \
    --cfg-scale 4.0 \
    --seed $SEED

# Move output to requested location
if [ -f "out.jpg" ]; then
    mv out.jpg "$OUTPUT"
    echo "Sample saved to: $OUTPUT"
else
    echo "Failed to generate sample"
fi