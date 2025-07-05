#!/bin/bash

echo "🎨 AI-Toolkit Generation Demo"
echo "============================"
echo ""

# Create output directory
mkdir -p outputs

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "❌ Error: Please run this script from the ai-toolkit-rs directory"
    exit 1
fi

echo "📦 Building examples..."
cargo build --examples --release 2>/dev/null

if [ $? -ne 0 ]; then
    echo "⚠️  Build failed, running in debug mode..."
    BINARY_PATH="target/debug/examples"
else
    BINARY_PATH="target/release/examples"
fi

echo ""
echo "1️⃣ Running quick generation demo..."
echo "This will generate sample images with SD1.5, SDXL, SD3.5, and Flux"
echo ""

if [ -f "$BINARY_PATH/quick_generate" ]; then
    $BINARY_PATH/quick_generate
else
    echo "⚠️  Quick generate example not built"
fi

echo ""
echo "2️⃣ Running video generation demo..."
echo "This will demonstrate video model generation"
echo ""

if [ -f "$BINARY_PATH/video_generation" ]; then
    $BINARY_PATH/video_generation
else
    echo "⚠️  Video generation example not built"
fi

echo ""
echo "3️⃣ For full model generation, run:"
echo "   cargo run --example generate_all_models -- --output-dir outputs"
echo ""
echo "4️⃣ For real model loading (requires model files), run:"
echo "   cargo run --example real_generation -- --model-dir /path/to/models --model sd35"
echo ""
echo "✅ Demo complete! Check the outputs directory for generated images."