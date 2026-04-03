#!/usr/bin/env python3
"""Cache Z-Image text embeddings — STRIPPED of padding tokens.

Diffusers/ai-toolkit strip padding via attention_mask before passing
to the model. We must do the same.
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

    # Get token IDs to find padding
    pad_id = encoder._tokenizer.pad_token_id
    if pad_id is None:
        pad_id = encoder._tokenizer.eos_token_id
    print(f"Pad token ID: {pad_id}")

    for label, prompt in [("pos", PROMPT), ("neg", NEGATIVE_PROMPT)]:
        token_ids = encoder._tokenizer.encode(prompt, add_special_tokens=True)
        # Pad to 512 like the encoder does
        padded = token_ids + [pad_id] * (512 - len(token_ids))
        mask = [1 if t != pad_id else 0 for t in padded]
        real_count = sum(mask)
        print(f"\n{label}: '{prompt}'")
        print(f"  Tokens: {len(token_ids)} real, {512 - len(token_ids)} padding")

    # Now encode and strip
    print("\nEncoding positive...")
    pos_output = encoder.encode(PROMPT)
    pos_full = pos_output.hidden_states  # [1, 512, 2560]

    # Get mask
    token_ids = encoder._tokenizer.encode(PROMPT, add_special_tokens=True)
    padded = token_ids + [pad_id] * (512 - len(token_ids))
    mask = torch.tensor([1 if t != pad_id else 0 for t in padded], dtype=torch.bool)
    pos_stripped = pos_full[0][mask].unsqueeze(0)  # [1, real_tokens, 2560]
    print(f"  Full: {pos_full.shape} → Stripped: {pos_stripped.shape}")

    print("Encoding negative...")
    neg_output = encoder.encode(NEGATIVE_PROMPT)
    neg_full = neg_output.hidden_states

    token_ids = encoder._tokenizer.encode(NEGATIVE_PROMPT, add_special_tokens=True)
    padded = token_ids + [pad_id] * (512 - len(token_ids))
    mask = torch.tensor([1 if t != pad_id else 0 for t in padded], dtype=torch.bool)
    neg_stripped = neg_full[0][mask].unsqueeze(0)
    print(f"  Full: {neg_full.shape} → Stripped: {neg_stripped.shape}")

    encoder.unload()
    del encoder
    torch.cuda.empty_cache()

    tensors = {
        "pos_hidden": pos_stripped.cpu().to(torch.bfloat16),
        "neg_hidden": neg_stripped.cpu().to(torch.bfloat16),
    }

    save_file(tensors, OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")
    for k, v in tensors.items():
        print(f"  {k}: {v.shape} {v.dtype}")

if __name__ == "__main__":
    main()
