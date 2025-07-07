#!/usr/bin/env python3
import safetensors
import sys

model_path = "/home/alex/SwarmUI/Models/diffusion_models/flux1-dev.safetensors"

# Open the file without loading tensors into memory
with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    
    print("Total tensors:", len(keys))
    print("\nMLP-related tensors (first 20):")
    
    mlp_keys = [k for k in keys if "mlp" in k]
    for i, key in enumerate(mlp_keys[:20]):
        print(f"  {key}")
    
    print("\nDouble block 0 tensors:")
    db0_keys = [k for k in keys if k.startswith("double_blocks.0.")]
    for key in sorted(db0_keys):
        print(f"  {key}")