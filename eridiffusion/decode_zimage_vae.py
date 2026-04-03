#!/usr/bin/env python3
"""Decode Z-Image latents from safetensors using Flux 1 VAE (ae.safetensors)."""
import sys
sys.path.insert(0, "/home/alex/serenity-inference")

import argparse
import torch
from safetensors.torch import load_file
from PIL import Image

VAE_PATH = "/home/alex/.serenity/models/vaes/ae.safetensors"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to latents safetensors")
    parser.add_argument("-o", "--output", default="/home/alex/serenity/output/zimage_rust.png")
    args = parser.parse_args()

    tensors = load_file(args.input)
    latents = tensors["latents"].to(dtype=torch.bfloat16, device="cuda")
    print(f"Loaded latents: {latents.shape} {latents.dtype}")
    print(f"Range: [{latents.min():.4f}, {latents.max():.4f}]")

    # Load Flux 1 VAE
    from engine.loader import load_state_dict
    vae_sd = load_state_dict(VAE_PATH)

    from models.vae import AutoencoderKL
    vae = AutoencoderKL.from_state_dict(vae_sd)
    del vae_sd
    vae = vae.to(device="cuda", dtype=torch.bfloat16)
    vae.eval()

    with torch.no_grad():
        decoded = vae.decode(latents)
    if isinstance(decoded, tuple):
        decoded = decoded[0]
    print(f"Decoded: {decoded.shape}")

    img_tensor = decoded[0].float().clamp(-1, 1)
    img_tensor = (img_tensor + 1.0) / 2.0
    img_tensor = img_tensor.permute(1, 2, 0)
    img_np = (img_tensor.cpu().numpy() * 255).astype("uint8")
    Image.fromarray(img_np).save(args.output)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
