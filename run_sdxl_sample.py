#!/usr/bin/env python3
"""
Quick test to generate SDXL image using diffusers library
This is just to verify the model works before implementing in Rust
"""

import torch
from diffusers import StableDiffusionXLPipeline

print("Loading SDXL pipeline...")
pipe = StableDiffusionXLPipeline.from_single_file(
    "/home/alex/SwarmUI/Models/diffusion_models/sd_xl_base_1.0.safetensors",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe = pipe.to("cuda")

print("Generating image: 'a white swan on mars'")
image = pipe(
    "a white swan on mars",
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]

output_path = "outputs/sdxl_lora/samples/sdxl_real_swan_python.jpg"
image.save(output_path)
print(f"Saved to: {output_path}")