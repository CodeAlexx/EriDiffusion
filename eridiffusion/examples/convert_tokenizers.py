#!/usr/bin/env python3
"""
Convert CLIP and T5 tokenizers to JSON format for use with the tokenizers crate.
"""

import json
import os
from transformers import AutoTokenizer

def convert_tokenizers():
    # Output directory
    output_dir = "/home/alex/diffusers-rs/eridiffusion/tokenizers"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Converting tokenizers to JSON format...")
    
    # Convert CLIP tokenizer
    print("\n1. Converting CLIP tokenizer...")
    clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_tokenizer.save_pretrained(os.path.join(output_dir, "clip"))
    print("   ✅ CLIP tokenizer saved")
    
    # Convert T5 tokenizer
    print("\n2. Converting T5-XXL tokenizer...")
    t5_tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-xxl")
    t5_tokenizer.save_pretrained(os.path.join(output_dir, "t5"))
    print("   ✅ T5 tokenizer saved")
    
    # Create a simple JSON wrapper for T5 if needed
    print("\n3. Creating JSON wrappers...")
    
    # For CLIP, the tokenizer.json should already exist
    clip_json_path = os.path.join(output_dir, "clip", "tokenizer.json")
    if os.path.exists(clip_json_path):
        print("   ✅ CLIP tokenizer.json exists")
    else:
        print("   ⚠️  CLIP tokenizer.json not found, using vocab.json")
    
    # For T5, we need to work with the sentencepiece model
    t5_spiece_path = os.path.join(output_dir, "t5", "spiece.model")
    if os.path.exists(t5_spiece_path):
        print("   ✅ T5 spiece.model exists")
        # The tokenizers crate can load sentencepiece models directly
        # But we'll create a simple wrapper config
        t5_config = {
            "type": "SentencePiece",
            "model_path": t5_spiece_path,
            "add_bos": False,
            "add_eos": True
        }
        with open(os.path.join(output_dir, "t5", "tokenizer_wrapper.json"), "w") as f:
            json.dump(t5_config, f, indent=2)
        print("   ✅ T5 tokenizer wrapper created")
    
    print("\n✅ Tokenizer conversion complete!")
    print(f"   Output directory: {output_dir}")
    print("\n   Files created:")
    print(f"   - CLIP: {output_dir}/clip/")
    print(f"   - T5: {output_dir}/t5/")

if __name__ == "__main__":
    convert_tokenizers()