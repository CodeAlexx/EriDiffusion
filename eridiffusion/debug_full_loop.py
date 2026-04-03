#!/usr/bin/env python3
"""Run full 8-step Z-Image sampling in Python using the SAME noise Rust uses.

Saves Rust noise → Python denoise → compare with Rust output.
"""
import sys
sys.path.insert(0, "/home/alex/serenity-inference")

import torch
from safetensors.torch import load_file, save_file

MODEL_PATH = "/home/alex/.serenity/models/checkpoints/z_image_de_turbo_v1_bf16.safetensors"
EMBEDDINGS_PATH = "/home/alex/EriDiffusion/eridiffusion/eridiffusion/cached_zimage_embeddings.safetensors"
RUST_LATENTS = "/home/alex/serenity/output/zimage_denoised_latents.safetensors"

device = torch.device("cuda")
dtype = torch.bfloat16

NUM_STEPS = 8
SHIFT = 3.0
CFG_SCALE = 4.0
SEED = 42

# -------------------------------------------------------------------
# Recreate EXACT same noise as Rust (Box-Muller with StdRng seed 42)
# Actually, easier: just generate with same seed and compare later
# Let's use Python torch noise instead and compare output quality
# -------------------------------------------------------------------

# Load embeddings
cached = load_file(EMBEDDINGS_PATH, device="cuda")
pos_hidden = cached["pos_hidden"].to(dtype=dtype)
neg_hidden = cached["neg_hidden"].to(dtype=dtype)

# Load model
from engine.loader import load_state_dict
from models.zimage_dit import NextDiT

model_sd = load_state_dict(MODEL_PATH)
model = NextDiT.from_state_dict(model_sd)
del model_sd
model = model.to(device=device, dtype=dtype)
model.eval()

# Sigma schedule (same as Rust)
def build_sigma_schedule(num_steps, shift=3.0):
    sigmas = []
    for i in range(num_steps + 1):
        t = 1.0 - (i / num_steps)
        if abs(shift - 1.0) > 1e-6:
            t = shift * t / (1.0 + (shift - 1.0) * t)
        sigmas.append(t)
    return sigmas

sigmas = build_sigma_schedule(NUM_STEPS, SHIFT)
print(f"Sigmas: {sigmas}")

# Create noise with PYTHON's RNG (different from Rust, but clean)
torch.manual_seed(SEED)
noise = torch.randn(1, 16, 64, 64, device=device, dtype=torch.float32).to(dtype=dtype)
print(f"Noise: {noise.shape}, range=[{noise.min():.4f}, {noise.max():.4f}]")

# Manual Euler loop (matching Rust exactly)
x = noise.clone()
for i in range(NUM_STEPS):
    sigma = sigmas[i]
    sigma_next = sigmas[i + 1]

    sigma_t = torch.tensor([sigma], device=device, dtype=dtype)

    with torch.no_grad():
        cond = model(x, sigma_t, pos_hidden)
        uncond = model(x, sigma_t, neg_hidden)

    # CFG
    guided = uncond + CFG_SCALE * (cond - uncond)

    # Denoised = x - guided * sigma
    denoised = x - guided * sigma

    # Euler step
    d = (x - denoised) / sigma  # = guided
    dt = sigma_next - sigma
    x = x + d * dt

    print(f"  Step {i+1}/{NUM_STEPS} sigma={sigma:.4f} range=[{x.min():.4f}, {x.max():.4f}]")

print(f"\nFinal: range=[{x.min():.4f}, {x.max():.4f}], std={x.float().std():.4f}")

# Decode
from PIL import Image
vae_sd = load_state_dict("/home/alex/.serenity/models/vaes/ae.safetensors")
from models.vae import AutoencoderKL
vae = AutoencoderKL.from_state_dict(vae_sd)
del vae_sd
vae = vae.to(device=device, dtype=dtype).eval()

with torch.no_grad():
    decoded = vae.decode(x)
if isinstance(decoded, tuple):
    decoded = decoded[0]

img = decoded[0].float().clamp(-1, 1)
img = ((img + 1) / 2).permute(1, 2, 0).cpu().numpy()
img = (img * 255).astype("uint8")
Image.fromarray(img).save("/home/alex/serenity/output/zimage_python_8step.png")
print("Saved: /home/alex/serenity/output/zimage_python_8step.png")

# Also compare with Rust output
rust_tensors = load_file(RUST_LATENTS, device="cuda")
rust_latents = rust_tensors["latents"].to(dtype=dtype)
print(f"\nRust latents: range=[{rust_latents.min():.4f}, {rust_latents.max():.4f}], std={rust_latents.float().std():.4f}")
print(f"Python latents: range=[{x.min():.4f}, {x.max():.4f}], std={x.float().std():.4f}")
diff = (x - rust_latents).float().abs()
print(f"Diff: max={diff.max():.4f}, mean={diff.mean():.6f}")
