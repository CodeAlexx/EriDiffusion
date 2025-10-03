//! Example of training with FLAME's mutable gradients

use eridiffusion::flame_training::{train_with_flame, FLAMEModel};
use flame::optim::AdamW;
use flame::{DType, Device, Parameter, Result, Tensor};

fn main() -> anyhow::Result<()> {
    let device = Device::cuda(0);

    // Create model
    let mut model = create_model(device)?;

    // Create optimizer
    let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8, 0.01);

    // Create dataloader
    let dataloader = create_dataloader()?;

    // Train with FLAME - mutable gradients work!
    let losses = train_with_flame(
        &mut model,
        dataloader,
        &mut optimizer,
        10, // epochs
    )?;

    println!("Training complete! Final loss: {}", losses.last().unwrap());

    Ok(())
}
