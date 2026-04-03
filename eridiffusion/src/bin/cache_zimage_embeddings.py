#!/usr/bin/env python3
"""Cache Z-Image text embeddings to safetensors for Rust inference.

One-time script — run once, then Rust binary loads the cached file.
"""
import sys
sys.path.insert(0, "/home/alex/serenity-inference")

import torch
from safetensors.torch import save_file

ENCODER_PATH = "/home/alex/.serenity/models/text_encoders/qwen_3_4b.safetensors"
OUTPUT_PATH = "/home/alex/EriDiffusion/eridiffusion/eridiffusion/cached_zimage_embeddings.safetensors"

PROMPT = "a photograph of an astronaut riding a horse on mars, cinematic lighting"
NEGATIVE_PROMPT = ""

def main():
    print("Loading Qwen3 4B encoder...")
    from text.qwen3 import Qwen3Encoder
    encoder = Qwen3Encoder(mode="zimage", dtype=torch.bfloat16, device="cuda")
    encoder.load(ENCODER_PATH)

    print(f"Encoding positive: {PROMPT}")
    pos_output = encoder.encode(PROMPT)
    pos_hidden = pos_output.hidden_states.to(dtype=torch.bfloat16)
    print(f"  pos_hidden: {pos_hidden.shape}")

    print(f"Encoding negative: '{NEGATIVE_PROMPT}'")
    neg_output = encoder.encode(NEGATIVE_PROMPT)
    neg_hidden = neg_output.hidden_states.to(dtype=torch.bfloat16)
    print(f"  neg_hidden: {neg_hidden.shape}")

    encoder.unload()
    del encoder
    torch.cuda.empty_cache()

    tensors = {
        "pos_hidden": pos_hidden.cpu(),
        "neg_hidden": neg_hidden.cpu(),
    }

    save_file(tensors, OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")
    for k, v in tensors.items():
        print(f"  {k}: {v.shape} {v.dtype}")

if __name__ == "__main__":
    main()
