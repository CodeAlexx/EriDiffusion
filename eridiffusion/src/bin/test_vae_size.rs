use anyhow::Result;
use eridiffusion::models::vae::{AutoEncoderKL, VAEConfig};
use flame_core::{DType, Device, Tensor};
use std::sync::Arc;

fn main() -> Result<()> {
    flame_core::init();
    let device = Device::cuda(0);

    println!("Loading VAE...");
    // We need to load weights from the checkpoint.
    // But for size testing, we don't strictly need valid weights, just the structure.
    // However, VAE::new requires config.
    // And we want to ensure it behaves exactly like the real one.
    
    // Let's define the config manually or load it.
    // SD3.5 VAE config:
    let config = VAEConfig {
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        latent_channels: 16,
        norm_num_groups: 32,
    };

    println!("Creating VAE...");
    let mut vae = AutoEncoderKL::new(&config, device.clone())?;

    // We don't need to load weights to check shapes.
    // But if we want to be sure, we could.
    // For now, let's just run forward with random weights (initialized in new).

    println!("Creating random latents...");
    let latents = Tensor::randn(
        flame_core::Shape::from_dims(&[1, 16, 64, 64]),
        0.0,
        1.0,
        device.clone(),
    )?.to_dtype(DType::BF16)?;

    println!("Decoding...");
    let image = vae.decode(&latents)?;

    println!("Output shape: {:?}", image.shape().dims());

    Ok(())
}
