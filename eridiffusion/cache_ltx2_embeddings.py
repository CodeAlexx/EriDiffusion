#!/usr/bin/env python3
"""Cache Gemma-3 12B text embeddings for LTX-2 Rust inference.

LTX-2 uses Gemma-3 12B (same as Z-Image) with penultimate layer output.
"""
import sys
sys.path.insert(0, "/home/alex/serenity-inference")

import torch
from safetensors.torch import save_file

ENCODER_PATH = "/home/alex/.serenity/models/text_encoders/qwen_3_4b.safetensors"
OUTPUT_PATH = "/home/alex/EriDiffusion/eridiffusion/eridiffusion/cached_ltx2_embeddings.safetensors"

PROMPT = "a cinematic dolly shot of an astronaut walking on mars, red dust, dramatic lighting, 4k"

def main():
    print("Loading Gemma/Qwen3 encoder (zimage mode = penultimate layer)...")
    from text.qwen3 import Qwen3Encoder
    encoder = Qwen3Encoder(mode="zimage", dtype=torch.bfloat16, device="cuda")
    encoder.load(ENCODER_PATH)

    print(f"Encoding: {PROMPT}")
    output = encoder.encode(PROMPT)
    text_hidden = output.hidden_states.to(dtype=torch.bfloat16)
    print(f"  text_hidden: {text_hidden.shape}")

    encoder.unload()
    del encoder
    torch.cuda.empty_cache()

    save_file({"text_hidden": text_hidden.cpu()}, OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
