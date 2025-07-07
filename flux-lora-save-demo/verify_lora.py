#!/usr/bin/env python3
from safetensors import safe_open

with safe_open("flux_lora_demo.safetensors", framework="pt", device="cpu") as f:
    keys = list(f.keys())
    print(f"Total tensors: {len(keys)}")
    print("\nTensor names:")
    for key in sorted(keys):
        tensor = f.get_tensor(key)
        print(f"  {key}: {list(tensor.shape)}")
    
    # Check metadata
    metadata = f.metadata()
    print("\nMetadata:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")