#!/usr/bin/env python3
import safetensors
from safetensors import safe_open
import os

lora_path = "/home/alex/SwarmUI/Models/diffusion_models/flux1-schnell.safetensors"

print(f"Analyzing: {lora_path}")

with safe_open(lora_path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    print(f"\nTotal tensors: {len(keys)}")
    
    # Group by patterns
    groups = {}
    for key in keys:
        parts = key.split('.')
        if len(parts) > 2:
            group = f"{parts[0]}.{parts[1]}"
            if group not in groups:
                groups[group] = []
            groups[group].append(key)
    
    print("\nLayer groups:")
    for group in sorted(groups.keys()):
        print(f"  {group}: {len(groups[group])} tensors")
    
    # Check attention structure
    to_q = [k for k in keys if "to_q" in k]
    to_k = [k for k in keys if "to_k" in k]
    to_v = [k for k in keys if "to_v" in k]
    qkv = [k for k in keys if "qkv" in k]
    
    print(f"\nAttention structure:")
    print(f"  to_q tensors: {len(to_q)}")
    print(f"  to_k tensors: {len(to_k)}")
    print(f"  to_v tensors: {len(to_v)}")
    print(f"  qkv tensors: {len(qkv)}")
    
    # Show first 10 tensors
    print("\nFirst 10 tensors:")
    for i, key in enumerate(keys[:10]):
        tensor = f.get_tensor(key)
        print(f"  {key}: {list(tensor.shape)}")
    
    # Show attention tensors
    print("\nAttention tensor examples:")
    attn_tensors = [k for k in keys if "attn" in k and ("weight" in k or "bias" in k)]
    for key in attn_tensors[:10]:
        tensor = f.get_tensor(key)
        print(f"  {key}: {list(tensor.shape)}")
    
    # Show double_blocks.0 structure
    print("\ndouble_blocks.0 tensors:")
    db0 = [k for k in keys if k.startswith("double_blocks.0.")]
    for key in db0:
        if "weight" in key or "bias" in key:
            print(f"  {key}")