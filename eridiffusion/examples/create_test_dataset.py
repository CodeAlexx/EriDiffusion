#!/usr/bin/env python3

import os
from PIL import Image
import numpy as np

# Create test dataset
os.makedirs("test_dataset", exist_ok=True)

# Create a few test images
for i in range(3):
    # Create random image
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    
    # Save image
    img_path = f"test_dataset/image_{i}.png"
    img.save(img_path)
    
    # Create caption
    caption = f"A test image number {i}"
    caption_path = f"test_dataset/image_{i}.txt"
    with open(caption_path, 'w') as f:
        f.write(caption)
    
    print(f"Created {img_path} and {caption_path}")

print("Test dataset created!")