#!/bin/bash
# Script to create a test dataset structure for Flux LoRA training

echo "Creating test dataset structure..."

# Create dataset directory
mkdir -p test_dataset

# Create sample caption files
cat > test_dataset/image1.txt << EOF
a photo of ohwx person smiling
EOF

cat > test_dataset/image2.txt << EOF
ohwx person wearing a hat in the park
EOF

cat > test_dataset/image3.txt << EOF
portrait of ohwx person, professional lighting
EOF

echo "Test dataset structure created!"
echo ""
echo "Expected structure:"
echo "test_dataset/"
echo "├── image1.jpg"
echo "├── image1.txt"
echo "├── image2.jpg"
echo "├── image2.txt"
echo "├── image3.jpg"
echo "└── image3.txt"
echo ""
echo "Note: You need to add actual image files (.jpg/.png) to the test_dataset folder"
echo "Each image should have a corresponding .txt file with the caption"