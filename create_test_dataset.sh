#!/bin/bash
# Create a minimal test dataset for Flux LoRA training

DATASET_DIR="/home/alex/diffusers-rs/test_dataset"

echo "Creating test dataset at: $DATASET_DIR"
mkdir -p "$DATASET_DIR"

# Create a simple test image using ImageMagick
if command -v convert &> /dev/null; then
    # Create 3 test images
    for i in 1 2 3; do
        convert -size 1024x1024 \
            -background white \
            -fill black \
            -gravity center \
            -pointsize 72 \
            label:"Test Image $i" \
            "$DATASET_DIR/test_$i.png"
        
        # Create corresponding caption files
        echo "a test image with number $i" > "$DATASET_DIR/test_$i.txt"
    done
    echo "✓ Created 3 test images with captions"
else
    echo "ImageMagick not found. Creating empty dataset structure..."
    touch "$DATASET_DIR/test_1.png"
    touch "$DATASET_DIR/test_1.txt"
    echo "a test image" > "$DATASET_DIR/test_1.txt"
fi

# Update the config file
CONFIG_PATH="/home/alex/diffusers-rs/eridiffusion/config/flux_lora_minimal.yaml"
sed -i "s|folder_path: \"/path/to/your/images\"|folder_path: \"$DATASET_DIR\"|g" "$CONFIG_PATH"

echo ""
echo "Dataset created at: $DATASET_DIR"
ls -la "$DATASET_DIR"
echo ""
echo "✓ Config file updated with dataset path"